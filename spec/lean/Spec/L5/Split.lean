import Spec.L2.Effects

/-!
# Layer 5: a split adds no expression node (`D-24`, `D-25`, `T-11`)
-/

namespace Spec

open Classical

/-- The selector that picks a column no selector of the list picks. -/
def notAny : List Selector → Selector
  | [] => .nameTest .always
  | P :: Ps => .and (.not P) (notAny Ps)

/-- A column passes `notAny` exactly when no selector of the list picks it. -/
theorem mem_notAny (Ps : List Selector) (c : Name × DType) :
    c ∈ (notAny Ps).denote ↔ ∀ P ∈ Ps, c ∉ P.denote := by
  induction Ps with
  | nil => simp [notAny, Selector.denote, Absence.denote]
  | cons P Ps ih => simp [notAny, Selector.denote, ih]

/-- `D-25`. The part a column goes to: the first part whose predicate picks it. -/
noncomputable def firstMatch (Ps : List Selector) (c : Name × DType) : Option ℕ :=
  Ps.findIdx? fun P => decide (c ∈ P.denote)

/-- `D-25`. The columns first-match binding sends to part `i`. -/
noncomputable def partOf (Ps : List Selector) (i : ℕ) (X : ColumnSet) : ColumnSet :=
  {c ∈ X | firstMatch Ps c = some i}

/-- `D-25`. The closed form of part `i`: its predicate, and none of the earlier ones. -/
def closedForm (Ps : List Selector) (i : ℕ) (h : i < Ps.length) : Selector :=
  .and Ps[i] (notAny (Ps.take i))

/-- `T-11`. First-match binding sends to part `i` exactly the columns that the closed form of part
`i` keeps: a part is an ordinary filter (`E-3`) of the whole remainder. -/
theorem T11.split_is_filter (Ps : List Selector) (i : ℕ) (h : i < Ps.length) (X : ColumnSet) :
    partOf Ps i X = (Effect.filtered (closedForm Ps i h)).apply X := by
  ext c
  simp only [partOf, firstMatch, Effect.apply, closedForm, Set.mem_inter_iff, Set.mem_ofPred_eq,
    Selector.denote, mem_notAny, List.findIdx?_eq_some_iff_getElem, decide_eq_true_eq]
  refine and_congr_right fun _ => ?_
  constructor
  · rintro ⟨_, hi, hj⟩
    refine ⟨hi, fun P hP => ?_⟩
    obtain ⟨j, hjlen, rfl⟩ := List.getElem_of_mem hP
    rw [List.getElem_take]
    exact hj j (by simp at hjlen; omega)
  · rintro ⟨hi, hj⟩
    refine ⟨h, hi, fun j hji => ?_⟩
    have hmem : Ps[j] ∈ Ps.take i := by
      rw [List.mem_iff_getElem]
      exact ⟨j, by simp; omega, by rw [List.getElem_take]⟩
    exact hj _ hmem

end Spec
