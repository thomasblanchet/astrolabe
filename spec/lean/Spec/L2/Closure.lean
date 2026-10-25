import Spec.L2.Effects

/-!
# Layer 2: closure and soundness of the record (`T-2`, `T-1`)
-/

namespace Spec

/-- `T-2`. Each effect's record rule keeps every column set the input record describes. -/
theorem T2.closure (ε : Effect) (r : Record) (X : ColumnSet) (h : describes r X) :
    describes (recordRule ε r) (ε.apply X) := by
  cases ε with
  | unchanged => exact h
  | emptied => intro n t hc; exact hc.elim
  | filtered C =>
    rintro n t ⟨hX, hC⟩
    obtain ⟨hn, ht⟩ := h n t hX
    refine ⟨?_, D14.surviving_sound C r.types n t hC ht⟩
    rintro (hP | hR)
    · exact hn hP
    · exact D14.certainly_rejects_sound C r.types n t hR ht hC
  | suffixed s =>
    rintro m t ⟨n, u, hX, heq⟩
    simp only [Prod.mk.injEq] at heq
    obtain ⟨hm, htu⟩ := heq
    subst hm htu
    obtain ⟨hn, ht⟩ := h _ _ hX
    refine ⟨?_, ht⟩
    rintro (hE | ⟨n', hn', hm⟩)
    · exact hE ⟨n, rfl⟩
    · exact hn (by rw [List.append_cancel_right hm]; exact hn')
  | prefixed p =>
    rintro m t ⟨n, u, hX, heq⟩
    simp only [Prod.mk.injEq] at heq
    obtain ⟨hm, htu⟩ := heq
    subst hm htu
    obtain ⟨hn, ht⟩ := h _ _ hX
    refine ⟨?_, ht⟩
    rintro (hS | ⟨n', hn', hm⟩)
    · exact hS ⟨n, rfl⟩
    · exact hn (by rw [List.append_cancel_left hm]; exact hn')
  | retyped U =>
    rintro m t ⟨n, u, hX, heq⟩
    simp only [Prod.mk.injEq] at heq
    obtain ⟨hm, htU⟩ := heq
    subst hm htU
    exact ⟨(h _ _ hX).1, rfl⟩

/-- `T-2` for `E-4`. The combined record keeps every union of column sets its operands describe. -/
theorem T2.closure_both (a b : Record) (X Y : ColumnSet) (ha : describes a X)
    (hb : describes b Y) : describes (bothRule a b) (X ∪ Y) := by
  rintro n t (hX | hY)
  · obtain ⟨hn, ht⟩ := ha n t hX
    exact ⟨fun h => hn h.1, Or.inl ht⟩
  · obtain ⟨hn, ht⟩ := hb n t hY
    exact ⟨fun h => hn h.2, Or.inr ht⟩

/-- `T-1`. The record of an expression describes what it denotes, under every binding whose
variables satisfy their declarations. -/
theorem T1.over_approximates {V : Type} (decl : V → Record) (β : V → ColumnSet)
    (hdecl : ∀ v, describes (decl v) (β v)) (e : RemExpr V) :
    describes (know decl e) (e.denote β) := by
  induction e with
  | var v => exact hdecl v
  | app ε e ih => exact T2.closure ε _ _ ih
  | both e₁ e₂ ih₁ ih₂ => exact T2.closure_both _ _ _ _ ih₁ ih₂

end Spec
