import Spec.L1.Selector

/-!
# Layer 1: what a selector settles without seeing a column (`D-14`, `surviving`)

An operation over a remainder cannot see a column. `certainlyPicks` and `certainlyRejects` give the
names a selector settles for every data type the remainder can hold, and `surviving` gives the data
types a filter can leave.
-/

namespace Spec

open Classical in
mutual
/-- `D-14`. The names whose every column the selector picks, for the data types in `Θ`. -/
noncomputable def certainlyPicks : Selector → TypeSet → Absence
  | .nameTest P, _ => P
  | .ofType F, Θ => if Θ.denote ⊆ F.denote then .always else .never
  | .not C, Θ => certainlyRejects C Θ
  | .and C D, Θ => .and (certainlyPicks C Θ) (certainlyPicks D Θ)
  | .or C D, Θ => .or (certainlyPicks C Θ) (certainlyPicks D Θ)

/-- `D-14`. The names whose every column the selector rejects, for the data types in `Θ`. -/
noncomputable def certainlyRejects : Selector → TypeSet → Absence
  | .nameTest P, _ => .not P
  | .ofType F, Θ => if Θ.denote ∩ F.denote = ∅ then .always else .never
  | .not C, Θ => certainlyPicks C Θ
  | .and C D, Θ => .or (certainlyRejects C Θ) (certainlyRejects D Θ)
  | .or C D, Θ => .and (certainlyRejects C Θ) (certainlyRejects D Θ)
end

/-- Section 7.1. The data types a filter by `C` can leave, from the data types in `Θ`. -/
def surviving : Selector → TypeSet → TypeSet
  | .nameTest _, Θ => Θ
  | .ofType F, Θ => .inter Θ (.ofSet F)
  | .not C, Θ => if C.readsNoName then .diff Θ (surviving C Θ) else Θ
  | .and C D, Θ => .inter (surviving C Θ) (surviving D Θ)
  | .or C D, Θ => .union (surviving C Θ) (surviving D Θ)

/-- Both halves of `D-14` at once: a name that `certainlyPicks` holds of is picked with every data
type in `Θ`, and one that `certainlyRejects` holds of is rejected with every such data type. -/
theorem D14.certainly_sound (C : Selector) (Θ : TypeSet) :
    (∀ n t, n ∈ (certainlyPicks C Θ).denote → t ∈ Θ.denote → (n, t) ∈ C.denote) ∧
    (∀ n t, n ∈ (certainlyRejects C Θ).denote → t ∈ Θ.denote → (n, t) ∉ C.denote) := by
  induction C with
  | nameTest P =>
    refine ⟨fun n t hn _ => ?_, fun n t hn _ hc => ?_⟩
    · simpa [certainlyPicks, Selector.denote] using hn
    · simp [certainlyRejects, Absence.denote, Selector.denote] at hn hc
      exact hn hc
  | ofType F =>
    refine ⟨fun n t hn ht => ?_, fun n t hn ht hc => ?_⟩
    · by_cases h : Θ.denote ⊆ F.denote
      · exact h ht
      · simp [certainlyPicks, h, Absence.denote] at hn
    · by_cases h : Θ.denote ∩ F.denote = ∅
      · have : t ∈ Θ.denote ∩ F.denote := ⟨ht, hc⟩
        rw [h] at this
        exact this
      · simp [certainlyRejects, h, Absence.denote] at hn
  | not C ih =>
    refine ⟨fun n t hn ht hc => ?_, fun n t hn ht hc => ?_⟩
    · exact ih.2 n t hn ht hc
    · exact hc (ih.1 n t hn ht)
  | and C D ihC ihD =>
    refine ⟨fun n t hn ht => ?_, fun n t hn ht hc => ?_⟩
    · exact ⟨ihC.1 n t hn.1 ht, ihD.1 n t hn.2 ht⟩
    · rcases hn with hn | hn
      · exact ihC.2 n t hn ht hc.1
      · exact ihD.2 n t hn ht hc.2
  | or C D ihC ihD =>
    refine ⟨fun n t hn ht => ?_, fun n t hn ht hc => ?_⟩
    · rcases hn with hn | hn
      · exact Or.inl (ihC.1 n t hn ht)
      · exact Or.inr (ihD.1 n t hn ht)
    · rcases hc with hc | hc
      · exact ihC.2 n t hn.1 ht hc
      · exact ihD.2 n t hn.2 ht hc

/-- `D-14`. A name `certainlyPicks` holds of is picked with every data type in `Θ`. -/
theorem D14.certainly_picks_sound (C : Selector) (Θ : TypeSet) (n : Name) (t : DType)
    (hn : n ∈ (certainlyPicks C Θ).denote) (ht : t ∈ Θ.denote) : (n, t) ∈ C.denote :=
  (Spec.D14.certainly_sound C Θ).1 n t hn ht

/-- `D-14`. A name `certainlyRejects` holds of is rejected with every data type in `Θ`. -/
theorem D14.certainly_rejects_sound (C : Selector) (Θ : TypeSet) (n : Name) (t : DType)
    (hn : n ∈ (certainlyRejects C Θ).denote) (ht : t ∈ Θ.denote) : (n, t) ∉ C.denote :=
  (Spec.D14.certainly_sound C Θ).2 n t hn ht

/-- For a selector that reads no name, `surviving` is exact: over the data types in `Θ`, a column
is picked exactly when its data type survives. -/
theorem D14.surviving_exact (C : Selector) (hC : C.readsNoName = true) (Θ : TypeSet)
    (n : Name) (t : DType) (ht : t ∈ Θ.denote) :
    (n, t) ∈ C.denote ↔ t ∈ (surviving C Θ).denote := by
  induction C with
  | nameTest P => simp [Selector.readsNoName] at hC
  | ofType F => simp [Selector.denote, surviving, TypeSet.denote, ht]
  | not C ih =>
    have hC' : C.readsNoName = true := by simpa [Selector.readsNoName] using hC
    simp [Selector.denote, surviving, hC', TypeSet.denote, ht, ih hC']
  | and C D ihC ihD =>
    simp [Selector.readsNoName] at hC
    simp [Selector.denote, surviving, TypeSet.denote, ihC hC.1, ihD hC.2]
  | or C D ihC ihD =>
    simp [Selector.readsNoName] at hC
    simp [Selector.denote, surviving, TypeSet.denote, ihC hC.1, ihD hC.2]

/-- Section 7.1. A column a filter by `C` keeps, whose data type is in `Θ`, has a data type in
`surviving C Θ`. The guard on the `not` row is what makes this hold. -/
theorem D14.surviving_sound (C : Selector) (Θ : TypeSet) (n : Name) (t : DType)
    (hc : (n, t) ∈ C.denote) (ht : t ∈ Θ.denote) : t ∈ (surviving C Θ).denote := by
  induction C with
  | nameTest P => simpa [surviving] using ht
  | ofType F => exact ⟨ht, hc⟩
  | not C ih =>
    by_cases hC : C.readsNoName = true
    · simp only [surviving, hC, ite_true, TypeSet.denote]
      exact ⟨ht, fun hs => hc ((Spec.D14.surviving_exact C hC Θ n t ht).2 hs)⟩
    · simpa [surviving, hC] using ht
  | and C D ihC ihD => exact ⟨ihC hc.1, ihD hc.2⟩
  | or C D ihC ihD =>
    rcases hc with hc | hc
    · exact Or.inl (ihC hc)
    · exact Or.inr (ihD hc)

end Spec
