use super::super::{LogicalNodeProcessor, ProcessableNode, PushDownBuilderContext};
use crate::logical_plan::{AggregateMultipliedSubquery, AggregateMultipliedSubquerySource};
use crate::physical_plan_builder::PhysicalPlanBuilder;
use crate::plan::{
    Expr, FilterItem, From, FromSource, Join, JoinCondition, JoinItem, MemberExpression,
    QualifiedColumnName, QueryPlan, Select, SelectBuilder, SingleAliasedSource,
};
use crate::plan::join::JoinType;
use crate::planner::sql_evaluator::ReferencesBuilder;
use cubenativeutils::CubeError;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

pub struct AggregateMultipliedSubqueryProcessor<'a> {
    builder: &'a PhysicalPlanBuilder,
}

impl<'a> AggregateMultipliedSubqueryProcessor<'a> {
    /// Extract filters that apply to a specific cube (by cube name).
    /// Returns a map of cube_name -> FilterItem containing only filters for that cube.
    fn partition_filters_by_cube(
        &self,
        all_filters: Option<FilterItem>,
        cube_names: &HashSet<String>,
    ) -> HashMap<String, FilterItem> {
        let mut result = HashMap::new();

        if let Some(filter_item) = all_filters {
            for cube_name in cube_names.iter() {
                if let Some(cube_filter) = filter_item.find_subtree_for_cube(cube_name) {
                    result.insert(cube_name.clone(), cube_filter);
                }
            }
        }

        result
    }
}

impl<'a> LogicalNodeProcessor<'a, AggregateMultipliedSubquery>
    for AggregateMultipliedSubqueryProcessor<'a>
{
    type PhysycalNode = Rc<Select>;
    fn new(builder: &'a PhysicalPlanBuilder) -> Self {
        Self { builder }
    }

    fn process(
        &self,
        aggregate_multiplied_subquery: &AggregateMultipliedSubquery,
        context: &PushDownBuilderContext,
    ) -> Result<Self::PhysycalNode, CubeError> {
        let query_tools = self.builder.query_tools();

        // If this is a dimensions-only query, process the keys subquery path
        if context.dimensions_query {
            let keys_query = self.builder.process_node(
                aggregate_multiplied_subquery.keys_subquery.as_ref(),
                context,
            )?;
            return Ok(keys_query);
        }

        // Direct build approach - no keys subquery + self-join pattern
        // Build FROM directly from the source with unrelated dimensions support
        let mut context_factory = context.make_sql_nodes_factory()?;

        let pk_cube = aggregate_multiplied_subquery
            .keys_subquery
            .pk_cube()
            .clone();
        let pk_cube_alias = pk_cube
            .cube()
            .default_alias_with_prefix(&context.alias_prefix);

        // Build FROM directly from the source (preserves unrelated dimensions handling)
        let mut updated_context = context.clone();
        updated_context.alias_prefix = Some(pk_cube_alias.clone());

        // Set up unrelated dimensions context with the combined schema
        // This ensures unrelated dimension detection works with all queried dimensions
        let mut cubes_with_dims_set = HashSet::new();
        let mut all_queried_dims = Vec::new();

        for dim in aggregate_multiplied_subquery.schema.all_dimensions() {
            let cube_name = dim.cube_name().to_string();
            cubes_with_dims_set.insert(cube_name.clone());
            all_queried_dims.push(dim.clone());
        }

        // Identify unrelated cubes (1=1 joins) to strip their filters from the outer WHERE
        let mut all_unrelated_cubes = HashSet::new();
        let mut queried_unrelated_cubes = HashSet::new();

        for join_item in aggregate_multiplied_subquery.keys_subquery.source().joins().iter() {
            if join_item.on_sql().is_unrelated_join_condition() {
                let cube_name = join_item.cube().name().to_string();
                all_unrelated_cubes.insert(cube_name.clone());
                if cubes_with_dims_set.contains(&cube_name) {
                    queried_unrelated_cubes.insert(cube_name);
                }
            }
        }

        updated_context.cubes_with_queried_dimensions = Some(cubes_with_dims_set);
        updated_context.queried_dimensions = Some(all_queried_dims);

        // Partition filters by cube name for unrelated join optimization
        // These get pushed into DISTINCT subqueries for queried unrelated cubes
        if let Some(filter) = aggregate_multiplied_subquery.keys_subquery.filter().all_filters() {
            let cube_filters = self.partition_filters_by_cube(filter.to_filter_item(), &queried_unrelated_cubes);
            if !cube_filters.is_empty() {
                updated_context.unrelated_cube_filters = Some(cube_filters);
            }
        }

        let from = match &aggregate_multiplied_subquery.source {
            AggregateMultipliedSubquerySource::Cube(_cube) => {
                // Direct build from the LogicalJoin source (preserves unrelated dimensions handling)
                // Process the keys_subquery source which contains the join structure
                self.builder.process_node(
                    aggregate_multiplied_subquery.keys_subquery.source().as_ref(),
                    &updated_context
                )?
            }
            AggregateMultipliedSubquerySource::MeasureSubquery(measure_subquery) => {
                // Process the LogicalJoin to get dimension structure (same as Cube path)
                let dimension_from = self.builder.process_node(
                    aggregate_multiplied_subquery.keys_subquery.source().as_ref(),
                    &updated_context,
                )?;

                // Process the measure subquery
                let subquery = self
                    .builder
                    .process_node(measure_subquery.as_ref(), &updated_context)?;

                // Set up measure references from the subquery
                for meas in aggregate_multiplied_subquery.schema.measures.iter() {
                    context_factory.add_ungrouped_measure_reference(
                        meas.full_name(),
                        QualifiedColumnName::new(
                            Some(pk_cube_alias.clone()),
                            subquery.schema().resolve_member_alias(meas),
                        ),
                    );
                }

                // Build join conditions on PK dimensions to connect measure subquery
                let primary_keys_dimensions = aggregate_multiplied_subquery
                    .keys_subquery
                    .primary_keys_dimensions();
                let conditions = primary_keys_dimensions
                    .iter()
                    .map(|dim| -> Result<_, CubeError> {
                        let pk_cube_expr = Expr::Member(MemberExpression::new(dim.clone()));
                        let alias_in_measure_subquery =
                            subquery.schema().resolve_member_alias(dim);
                        let measure_subquery_ref = Expr::Reference(QualifiedColumnName::new(
                            Some(pk_cube_alias.clone()),
                            alias_in_measure_subquery,
                        ));
                        Ok(vec![(pk_cube_expr, measure_subquery_ref)])
                    })
                    .collect::<Result<Vec<_>, _>>()?;

                // Extend the dimension FROM with the measure subquery as a LEFT JOIN
                let measure_join_item = JoinItem {
                    from: SingleAliasedSource::new_from_subquery(
                        Rc::new(QueryPlan::Select(subquery)),
                        pk_cube_alias.clone(),
                    ),
                    on: JoinCondition::new_dimension_join(conditions, false),
                    join_type: JoinType::Left,
                };

                match &dimension_from.source {
                    FromSource::Join(join) => {
                        let mut new_joins = join.joins.clone();
                        new_joins.push(measure_join_item);
                        From::new_from_join(Rc::new(Join {
                            root: join.root.clone(),
                            joins: new_joins,
                        }))
                    }
                    FromSource::Single(source) => From::new_from_join(Rc::new(Join {
                        root: source.clone(),
                        joins: vec![measure_join_item],
                    })),
                    _ => {
                        return Err(CubeError::internal(
                            "Expected Single or Join from LogicalJoin processing".to_string(),
                        ));
                    }
                }
            }
        };

        let references_builder = ReferencesBuilder::new(from.clone());
        let mut select_builder = SelectBuilder::new(from.clone());
        let mut group_by = Vec::new();

        self.builder.resolve_subquery_dimensions_references(
            &aggregate_multiplied_subquery.dimension_subqueries,
            &references_builder,
            &mut context_factory,
        )?;

        // Add dimensions to SELECT and GROUP BY
        for member in aggregate_multiplied_subquery.schema.all_dimensions() {
            references_builder.resolve_references_for_member(
                member.clone(),
                &None,
                context_factory.render_references_mut(),
            )?;
            let alias = references_builder.resolve_alias_for_member(&member, &None);
            group_by.push(Expr::Member(MemberExpression::new(member.clone())));
            select_builder.add_projection_member(&member, alias);
        }

        // Add measures to SELECT
        for (measure, exists) in self
            .builder
            .measures_for_query(&aggregate_multiplied_subquery.schema.measures, &context)
        {
            if exists {
                if matches!(
                    &aggregate_multiplied_subquery.source,
                    AggregateMultipliedSubquerySource::Cube(_)
                ) {
                    references_builder.resolve_references_for_member(
                        measure.clone(),
                        &None,
                        context_factory.render_references_mut(),
                    )?;
                }
                select_builder.add_projection_member(&measure, None);
            } else {
                select_builder.add_null_projection(&measure, None);
            }
        }

        // Set GROUP BY - this replaces the DISTINCT in the keys subquery
        // GROUP BY achieves the same deduplication as SELECT DISTINCT + self-join, but in one pass
        select_builder.set_group_by(group_by);

        // Apply filters from the keys subquery at the outer SELECT level,
        // stripping filters for all unrelated cubes
        let filter = aggregate_multiplied_subquery.keys_subquery.filter().all_filters();
        let filter = if !all_unrelated_cubes.is_empty() {
            filter.and_then(|f| f.remove_filters_for_cubes(&all_unrelated_cubes))
        } else {
            filter
        };
        select_builder.set_filter(filter);

        context_factory.set_rendered_as_multiplied_measures(
            aggregate_multiplied_subquery
                .schema
                .multiplied_measures
                .clone(),
        );

        Ok(Rc::new(
            select_builder.build(query_tools.clone(), context_factory),
        ))
    }
}

impl ProcessableNode for AggregateMultipliedSubquery {
    type ProcessorType<'a> = AggregateMultipliedSubqueryProcessor<'a>;
}
