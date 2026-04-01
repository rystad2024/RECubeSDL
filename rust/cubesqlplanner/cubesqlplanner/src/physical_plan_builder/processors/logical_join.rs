use super::super::{LogicalNodeProcessor, ProcessableNode, PushDownBuilderContext};
use crate::logical_plan::LogicalJoin;
use crate::physical_plan_builder::PhysicalPlanBuilder;
use crate::plan::{From, JoinBuilder, JoinCondition, SelectBuilder};
use crate::planner::base_join_condition::AlwaysTrueJoinCondition;
use crate::planner::SqlJoinCondition;
use crate::planner::sql_evaluator::MemberSymbol;
use cubenativeutils::CubeError;
use std::rc::Rc;

pub struct LogicalJoinProcessor<'a> {
    builder: &'a PhysicalPlanBuilder,
}

impl<'a> LogicalNodeProcessor<'a, LogicalJoin> for LogicalJoinProcessor<'a> {
    type PhysycalNode = Rc<From>;
    fn new(builder: &'a PhysicalPlanBuilder) -> Self {
        Self { builder }
    }

    fn process(
        &self,
        logical_join: &LogicalJoin,
        context: &PushDownBuilderContext,
    ) -> Result<Self::PhysycalNode, CubeError> {
        let multi_stage_dimension = context.get_multi_stage_dimensions()?;
        if logical_join.root().is_none() {
            let res = if let Some(multi_stage_dimension) = &multi_stage_dimension {
                From::new_from_table_reference(
                    multi_stage_dimension.name.clone(),
                    multi_stage_dimension.schema.clone(),
                    None,
                )
            } else {
                From::new_empty()
            };
            return Ok(res);
        }

        let root = logical_join.root().clone().unwrap().cube().clone();
        if logical_join.joins().is_empty()
            && logical_join.dimension_subqueries().is_empty()
            && multi_stage_dimension.is_none()
        {
            Ok(From::new_from_cube(
                root.clone(),
                Some(root.default_alias_with_prefix(&context.alias_prefix)),
            ))
        } else {
            let mut join_builder = JoinBuilder::new_from_cube(
                root.clone(),
                Some(root.default_alias_with_prefix(&context.alias_prefix)),
            );

            for dimension_subquery in logical_join
                .dimension_subqueries() //TODO move dimension_subquery to
                .iter()
                .filter(|d| &d.subquery_dimension.cube_name() == root.name())
            {
                self.builder.add_subquery_join(
                    dimension_subquery.clone(),
                    &mut join_builder,
                    context,
                )?;
            }
            for join in logical_join.joins().iter() {
                let cube_name = join.cube().cube().name();
                let is_unrelated = join.on_sql().is_unrelated_join_condition();

                // Skip unrelated joins if no dimensions from this cube are queried
                if is_unrelated {
                    if let Some(ref queried_cubes) = context.cubes_with_queried_dimensions {
                        if !queried_cubes.contains(cube_name) {
                            // Skip this join entirely - no dimensions from this cube are needed
                            continue;
                        }
                    }
                }

                // For unrelated joins that ARE needed, create a DISTINCT subquery
                if is_unrelated && context.queried_dimensions.is_some() {
                    let queried_dims = context.queried_dimensions.as_ref().unwrap();
                    let cube_alias = join.cube()
                        .cube()
                        .default_alias_with_prefix(&context.alias_prefix);

                    // Create DISTINCT subquery with only the dimensions from this cube
                    let distinct_subquery = self.create_distinct_subquery_for_unrelated_cube(
                        join.cube().cube(),
                        &cube_alias,
                        context,
                        queried_dims,
                    )?;

                    // Join the subquery instead of the cube directly
                    let subquery_alias = format!("{}_distinct", cube_alias);
                    join_builder.left_join_subselect(
                        distinct_subquery,
                        subquery_alias,
                        JoinCondition::new_base_join(AlwaysTrueJoinCondition::new()),
                    );
                } else {
                    // Normal join handling
                    let join_condition = if is_unrelated {
                        JoinCondition::new_base_join(AlwaysTrueJoinCondition::new())
                    } else {
                        JoinCondition::new_base_join(SqlJoinCondition::try_new(join.on_sql().clone())?)
                    };

                    join_builder.left_join_cube(
                        join.cube().cube().clone(),
                        Some(
                            join.cube()
                                .cube()
                                .default_alias_with_prefix(&context.alias_prefix),
                        ),
                        join_condition,
                    );
                }

                for dimension_subquery in logical_join
                    .dimension_subqueries()
                    .iter()
                    .filter(|d| &d.subquery_dimension.cube_name() == join.cube().cube().name())
                {
                    self.builder.add_subquery_join(
                        dimension_subquery.clone(),
                        &mut join_builder,
                        context,
                    )?;
                }
            }
            if let Some(multi_stage_dimension) = &multi_stage_dimension {
                self.builder.add_multistage_dimension_join(
                    multi_stage_dimension,
                    &mut join_builder,
                    &context,
                )?;
            }
            Ok(From::new_from_join(join_builder.build()))
        }
    }
}

impl<'a> LogicalJoinProcessor<'a> {
    /// Create a DISTINCT subquery for unrelated dimension joins.
    /// This subquery selects only the distinct dimension values from the cube.
    fn create_distinct_subquery_for_unrelated_cube(
        &self,
        cube: &Rc<crate::planner::BaseCube>,
        cube_alias: &str,
        context: &PushDownBuilderContext,
        queried_dimensions: &[Rc<MemberSymbol>],
    ) -> Result<Rc<crate::plan::Select>, CubeError> {
        // Create FROM for the cube
        let from = From::new_from_cube(cube.clone(), Some(cube_alias.to_string()));
        let mut select_builder = SelectBuilder::new(from);

        // Add all queried dimensions from this cube to the SELECT DISTINCT
        for dim in queried_dimensions {
            if dim.cube_name() == *cube.name() {
                select_builder.add_projection_member(dim, None);
            }
        }

        // Set DISTINCT flag
        select_builder.set_distinct();

        // Apply filters specific to this cube if available
        if let Some(ref cube_filters) = context.unrelated_cube_filters {
            if let Some(filter_item) = cube_filters.get(cube.name()) {
                let filter = crate::plan::Filter {
                    items: vec![filter_item.clone()],
                };
                select_builder.set_filter(Some(filter));
            }
        }

        // Build the select statement
        let context_factory = context.make_sql_nodes_factory()?;
        Ok(Rc::new(select_builder.build(
            self.builder.query_tools().clone(),
            context_factory,
        )))
    }
}

impl ProcessableNode for LogicalJoin {
    type ProcessorType<'a> = LogicalJoinProcessor<'a>;
}
