use super::sql_evaluator::SqlCall;
use super::{evaluate_sql_call_with_context, VisitorContext};
use crate::planner::sql_templates::PlanSqlTemplates;
use cubenativeutils::CubeError;
use std::rc::Rc;
pub trait BaseJoinCondition {
    fn to_sql(
        &self,
        context: Rc<VisitorContext>,
        templates: &PlanSqlTemplates,
    ) -> Result<String, CubeError>;
}
pub struct SqlJoinCondition {
    sql_call: Rc<SqlCall>,
}
impl SqlJoinCondition {
    pub fn try_new(sql_call: Rc<SqlCall>) -> Result<Rc<Self>, CubeError> {
        Ok(Rc::new(Self { sql_call }))
    }
}

impl BaseJoinCondition for SqlJoinCondition {
    fn to_sql(
        &self,
        context: Rc<VisitorContext>,
        templates: &PlanSqlTemplates,
    ) -> Result<String, CubeError> {
        evaluate_sql_call_with_context(&self.sql_call, context, templates)
    }
}

/// A join condition that always evaluates to true, used for unrelated dimension joins (cross joins).
/// This is used when joining cubes that have no actual relationship (indicated by "1 = 1" join conditions).
pub struct AlwaysTrueJoinCondition;

impl AlwaysTrueJoinCondition {
    pub fn new() -> Rc<Self> {
        Rc::new(Self)
    }
}

impl BaseJoinCondition for AlwaysTrueJoinCondition {
    fn to_sql(
        &self,
        _context: Rc<VisitorContext>,
        templates: &PlanSqlTemplates,
    ) -> Result<String, CubeError> {
        templates.always_true()
    }
}
