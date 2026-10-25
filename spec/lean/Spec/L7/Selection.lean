import Spec.L2.Closure

/-!
# Layer 7: a selection materializes as the frame effects (`D-38`, `T-20`)
-/

namespace Spec

/-- `D-38`. The admissible name rewrites of a selection. -/
inductive Rename
  | none
  | suffixWith (s : Name)
  | prefixWith (p : Name)

/-- What a rewrite does to one name. -/
def Rename.fn : Rename → Name → Name
  | .none, n => n
  | .suffixWith s, n => n ++ s
  | .prefixWith p, n => p ++ n

/-- The effect of a rewrite on the remainder: `E-5a`, `E-5b`, or none. -/
def Rename.effect : Rename → Effect
  | .none => .unchanged
  | .suffixWith s => .suffixed s
  | .prefixWith p => .prefixed p

theorem Rename.effect_apply (ρ : Rename) (X : ColumnSet) :
    ρ.effect.apply X = {c | ∃ n t, (n, t) ∈ X ∧ c = (ρ.fn n, t)} := by
  cases ρ with
  | none =>
    ext ⟨n, t⟩
    simp [Rename.effect, Rename.fn, Effect.apply]
  | suffixWith s => rfl
  | prefixWith p => rfl

/-- `T-20`. What a selection with selector `C`, per-column table `u` and rewrite `ρ` gives from
the remainder `X`: each picked column at which `u` answers, renamed. -/
def selectFrom (C : Selector) (u : DType → Option DType) (ρ : Rename) (X : ColumnSet) :
    ColumnSet :=
  {c | ∃ n t t', (n, t) ∈ X ∧ (n, t) ∈ C.denote ∧ u t = some t' ∧ c = (ρ.fn n, t')}

/-- `T-20`. A selection whose table gives each column back, or gives one fixed data type,
materializes on the remainder as the chain of frame effects `Only`, then `Retyped`, then the
rewrite. -/
theorem T20.selection_as_effects (C : Selector) (ρ : Rename) (X : ColumnSet) :
    selectFrom C some ρ X = ρ.effect.apply ((Effect.filtered C).apply X) ∧
      ∀ U, selectFrom C (fun _ => some U) ρ X =
        ρ.effect.apply ((Effect.retyped U).apply ((Effect.filtered C).apply X)) := by
  refine ⟨?_, fun U => ?_⟩
  · ext ⟨m, t'⟩
    rw [Rename.effect_apply]
    simp only [selectFrom, Effect.apply, Set.mem_ofPred_eq,
      Set.mem_inter_iff, Option.some.injEq]
    constructor
    · rintro ⟨n, t, _, hX, hC, rfl, heq⟩
      exact ⟨n, t, ⟨hX, hC⟩, heq⟩
    · rintro ⟨n, t, ⟨hX, hC⟩, heq⟩
      exact ⟨n, t, t, hX, hC, rfl, heq⟩
  · ext ⟨m, t'⟩
    rw [Rename.effect_apply]
    simp only [selectFrom, Effect.apply, Set.mem_ofPred_eq,
      Set.mem_inter_iff, Option.some.injEq, Prod.mk.injEq]
    constructor
    · rintro ⟨n, t, t₁, hX, hC, hU, hm, ht'⟩
      exact ⟨n, U, ⟨n, t, ⟨hX, hC⟩, rfl, rfl⟩, hm, ht'.trans hU.symm⟩
    · rintro ⟨n, t, ⟨n₁, t₁, ⟨hX, hC⟩, hn, ht⟩, hm, ht'⟩
      exact ⟨n₁, t₁, U, hX, hC, rfl, hm.trans (by rw [hn]), ht'.trans ht⟩

/-- `T-20`. For any other table, the record of the effect chain still gives the names, and the
data types become `Unchecked`: every type is possible. -/
theorem T20.selection_approx (C : Selector) (u : DType → Option DType) (ρ : Rename) (r : Record)
    (X : ColumnSet) (h : describes r X) :
    describes ⟨(recordRule ρ.effect (recordRule (.filtered C) r)).absent, .univ⟩
      (selectFrom C u ρ X) := by
  rintro m t' ⟨n, t, t'', hX, hC, _, heq⟩
  have hchain := T2.closure ρ.effect _ _ (T2.closure (.filtered C) r X h)
  have hmem : (m, t) ∈ ρ.effect.apply ((Effect.filtered C).apply X) := by
    rw [Rename.effect_apply]
    rw [Prod.mk.injEq] at heq
    exact ⟨n, t, ⟨hX, hC⟩, by rw [heq.1]⟩
  exact ⟨(hchain m t hmem).1, trivial⟩

/-- `T-20`, beside other columns. A selection that also reads the columns `N` has a table `v` of the
matched column's data type and of the data types `ts` of the frame's columns. At one frame `ts` is
fixed, so the table is one function of the matched column's data type, and the approximation holds
as for any table. -/
theorem T20.selection_beside_approx (C : Selector) (v : DType → (Name → DType) → Option DType)
    (ts : Name → DType) (ρ : Rename) (r : Record) (X : ColumnSet) (h : describes r X) :
    describes ⟨(recordRule ρ.effect (recordRule (.filtered C) r)).absent, .univ⟩
      (selectFrom C (fun t => v t ts) ρ X) :=
  T20.selection_approx C (fun t => v t ts) ρ r X h

end Spec
