import Spec
import Lean

/-!
# Axiom audit

Every theorem that `spec/OPERATIONS.md` cites is listed here. The build fails when one of them
depends on an axiom other than Lean's three standard ones, which catches `sorry`, `native_decide`
and any added `axiom`.
-/

open Lean Elab Command

/-- Fail unless the named theorem uses only `propext`, `Classical.choice` and `Quot.sound`. -/
elab "#assert_standard_axioms " id:ident : command => do
  let name ← liftCoreM <| realizeGlobalConstNoOverloadWithInfo id
  let used ← collectAxioms name
  let allowed : List Lean.Name := [``propext, ``Classical.choice, ``Quot.sound]
  let bad := used.toList.filter fun a => !allowed.contains a
  unless bad.isEmpty do
    throwError m!"{name} depends on non-standard axioms: {bad}"

#assert_standard_axioms Spec.Functional.mono
#assert_standard_axioms Spec.D14.certainly_picks_sound
#assert_standard_axioms Spec.D14.certainly_rejects_sound
#assert_standard_axioms Spec.D14.surviving_exact
#assert_standard_axioms Spec.D14.surviving_sound
#assert_standard_axioms Spec.E4.functional_union_iff
#assert_standard_axioms Spec.T2.closure
#assert_standard_axioms Spec.T2.closure_both
#assert_standard_axioms Spec.T1.over_approximates
#assert_standard_axioms Spec.T3.absent_iff
#assert_standard_axioms Spec.T3.admissible_closed
#assert_standard_axioms Spec.T3.functional_rename
#assert_standard_axioms Spec.T3.collision
#assert_standard_axioms Spec.T3.suffix_admissible
#assert_standard_axioms Spec.T3.prefix_admissible
#assert_standard_axioms Spec.D23.translate_correct
#assert_standard_axioms Spec.T4.canonical_form_unique
#assert_standard_axioms Spec.T5.normalize_preserves_denotation
#assert_standard_axioms Spec.T8.equal_forms_equal_denotations
#assert_standard_axioms Spec.D11.regular
#assert_standard_axioms Spec.D11.preimage_regular
#assert_standard_axioms Spec.T9.witness_exists
#assert_standard_axioms Spec.T9.reduction
#assert_standard_axioms Spec.T9.regular
#assert_standard_axioms Spec.T9.finite_check
#assert_standard_axioms Spec.T9.decidable
#assert_standard_axioms Spec.T12.holds_iff_allowed
#assert_standard_axioms Spec.T12.decide_iff
#assert_standard_axioms Spec.T12.disjoint_iff
#assert_standard_axioms Spec.T12.within_row_unsound
#assert_standard_axioms Spec.T12.within_iff
#assert_standard_axioms Spec.T13.disjoint_union_iff
#assert_standard_axioms Spec.T13.subset_union_of_subset
#assert_standard_axioms Spec.T13.subset_union_not_imp
#assert_standard_axioms Spec.T11.split_is_filter
#assert_standard_axioms Spec.T14.sound
#assert_standard_axioms Spec.T14.requirement_sound
#assert_standard_axioms Spec.T20.selection_as_effects
#assert_standard_axioms Spec.T20.selection_approx
#assert_standard_axioms Spec.T20.selection_beside_approx
#assert_standard_axioms Spec.subset_collapse
#assert_standard_axioms Spec.collapse_mono
#assert_standard_axioms Spec.T23.fit_sound
#assert_standard_axioms Spec.T23.fit_sound_exact
#assert_standard_axioms Spec.T23.unchecked_counterexample
#assert_standard_axioms Spec.D22.distribute_denote
#assert_standard_axioms Spec.Node.T25.evaluate_eq_answer
#assert_standard_axioms Spec.Node.D41.lift_sound
#assert_standard_axioms Spec.Node.T22.monotone
#assert_standard_axioms Spec.Node.T22.no_new_error
#assert_standard_axioms Spec.Node.T21.containment
#assert_standard_axioms Spec.T21.unnest_counterexample
#assert_standard_axioms Spec.Generalize.fits_join_left
#assert_standard_axioms Spec.Generalize.fits_join_right
#assert_standard_axioms Spec.Generalize.fits_generalize
