import Spec.L0.Objects

/-!
# Layer 7: the contexts, the row forms of a node, and the cells of a declaration (`D-28`, `D-29`)

A frame operation evaluates an expression in one of two contexts. In the aggregation context an
expression that gives one value per row gives a `List` of its row data type (`T-15`). A declaration
states what an expression gives for the kinds of the columns that it reads, cell by cell.
-/

namespace Spec

/-- `D-28`. The two evaluation contexts. -/
inductive Context
  | row
  | aggregation

/-- `D-29`. The row form of an expression: one value in all, one value per group, or one value per
row. -/
inductive Character
  | literal
  | aggregating
  | elementwise
  deriving DecidableEq

/-- `T-15`. The context rule: in the aggregation context an expression that gives one value per row
gives a `List` of its row data type. It is measured, and this model takes it as the meaning of the
aggregation context. -/
def contextRule : Context → Character → DType → DType
  | .aggregation, .elementwise, t => .list t
  | _, _, t => t

/-- `D-39`. A cell of a declaration: the kind of each column that it reads. -/
abbrev Cell := Name → Kind

/-- `D-32`. The output names of `.struct.unnest()`: the fields of the struct it reads, unknown for
any other data type. -/
def unnestNames : DType → Option (List Name)
  | .struct fields => some (fields.map Prod.fst)
  | _ => none

/-- `T-21`. The exception: the output names of `.struct.unnest()` depend on the data type of the
column it reads, so replacing that data type by `Unchecked` changes names, not only data types. -/
theorem T21.unnest_counterexample :
    ∃ t₁ t₂ : DType, t₁.kind = t₂.kind ∧ unnestNames t₁ ≠ unnestNames t₂ := by
  refine ⟨.struct [(['a'], .scalar .int64)], .struct [(['b'], .scalar .int64)], rfl, ?_⟩
  simp [unnestNames]

end Spec
