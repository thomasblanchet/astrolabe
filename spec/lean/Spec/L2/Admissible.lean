import Spec.L2.Effects

/-!
# Layer 2: when a new effect can be added (`T-3`)
-/

namespace Spec

/-- An effect that renames every column by `f` and keeps its data type. -/
def rename (f : Name → Name) (X : ColumnSet) : ColumnSet :=
  {c | ∃ n t, (n, t) ∈ X ∧ c = (f n, t)}

/-- `T-3`. A renaming is admissible when it sends no two names to one, and the preimage of every
absence rule is an absence rule. -/
def Admissible (f : Name → Name) : Prop :=
  Function.Injective f ∧ ∀ P : Absence, ∃ Q : Absence, Q.denote = {m | ∀ n, f n = m → n ∈ P.denote}

/-- `T-3`, the displayed equivalence. A name is absent from the renamed column set exactly when
every name `f` sends to it is absent from the input. -/
theorem T3.absent_iff (f : Name → Name) (X : ColumnSet) (m : Name) :
    (∀ t, (m, t) ∉ rename f X) ↔ ∀ n, f n = m → ∀ t, (n, t) ∉ X := by
  constructor
  · intro h n hn t hX
    exact h t ⟨n, t, hX, by rw [hn]⟩
  · rintro h t ⟨n, u, hX, heq⟩
    simp only [Prod.mk.injEq] at heq
    obtain ⟨hm, htu⟩ := heq
    subst htu
    exact h n hm.symm t hX

/-- `T-3`. An admissible renaming has a sound record rule: the preimage of the absence rule. -/
theorem T3.admissible_closed (f : Name → Name) (hf : Admissible f) (r : Record) :
    ∃ r' : Record, ∀ X, describes r X → describes r' (rename f X) := by
  obtain ⟨Q, hQ⟩ := hf.2 r.absent
  refine ⟨⟨Q, r.types⟩, fun X h => ?_⟩
  rintro m t ⟨n, u, hX, heq⟩
  simp only [Prod.mk.injEq] at heq
  obtain ⟨hm, htu⟩ := heq
  subst hm htu
  obtain ⟨hn, ht⟩ := h _ _ hX
  refine ⟨fun hm => hn ?_, ht⟩
  rw [hQ] at hm
  exact hm n rfl

/-- `T-3`. An injective renaming keeps a column set functional. -/
theorem T3.functional_rename (f : Name → Name) (hf : Function.Injective f) (X : ColumnSet)
    (hX : Functional X) : Functional (rename f X) := by
  rintro m t u ⟨n, t', hn, heq⟩ ⟨n', u', hn', heq'⟩
  simp only [Prod.mk.injEq] at heq heq'
  obtain ⟨hm, htt⟩ := heq
  obtain ⟨hm', huu⟩ := heq'
  subst htt huu
  have hnn : n = n' := hf (hm.symm.trans hm')
  subst hnn
  exact hX n t u hn hn'

/-- `T-3`. A renaming that sends two names to one breaks a column set: two columns of different
data types can land on one name. -/
theorem T3.collision (f : Name → Name) (hf : ¬ Function.Injective f) :
    ∃ X : ColumnSet, Functional X ∧ ¬ Functional (rename f X) := by
  obtain ⟨a, b, hab, hne⟩ : ∃ a b, f a = f b ∧ a ≠ b := by
    simp only [Function.Injective, not_forall, exists_prop] at hf
    exact hf
  refine ⟨{(a, .scalar .int64), (b, .scalar .string)}, ?_, ?_⟩
  · intro n t u ht hu
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff, Prod.mk.injEq] at ht hu
    rcases ht with ⟨rfl, rfl⟩ | ⟨rfl, rfl⟩ <;> rcases hu with ⟨h1, rfl⟩ | ⟨h1, rfl⟩ <;> simp_all
  · intro h
    have := h (f a) (.scalar .int64) (.scalar .string) ⟨a, _, Or.inl rfl, rfl⟩
      ⟨b, _, Or.inr rfl, by rw [hab]⟩
    cases this

/-- `T-3`. Appending a suffix is admissible: its preimage rule is the `E-5a` rule. -/
theorem T3.suffix_admissible (s : Name) : Admissible (· ++ s) := by
  refine ⟨fun a b h => List.append_cancel_right h, fun P => ?_⟩
  refine ⟨.or (.not (.endsWith s)) (.afterStripping s P), ?_⟩
  ext m
  simp only [Absence.denote, Set.mem_union, Set.mem_compl_iff, Set.mem_ofPred_eq]
  constructor
  · rintro (hE | ⟨k, hk, rfl⟩) n hn
    · exact absurd ⟨n, hn⟩ hE
    · exact (List.append_cancel_right hn) ▸ hk
  · intro h
    by_cases hE : s <:+ m
    · obtain ⟨k, rfl⟩ := hE
      exact Or.inr ⟨k, h k rfl, rfl⟩
    · exact Or.inl hE

/-- `T-3`. Prepending a prefix is admissible: its preimage rule is the `E-5b` rule. -/
theorem T3.prefix_admissible (p : Name) : Admissible (p ++ ·) := by
  refine ⟨fun a b h => List.append_cancel_left h, fun P => ?_⟩
  refine ⟨.or (.not (.startsWith p)) (.beforeStripping p P), ?_⟩
  ext m
  simp only [Absence.denote, Set.mem_union, Set.mem_compl_iff, Set.mem_ofPred_eq]
  constructor
  · rintro (hS | ⟨k, hk, rfl⟩) n hn
    · exact absurd ⟨n, hn⟩ hS
    · exact (List.append_cancel_left hn) ▸ hk
  · intro h
    by_cases hS : p <+: m
    · obtain ⟨k, rfl⟩ := hS
      exact Or.inr ⟨k, h k rfl, rfl⟩
    · exact Or.inl hS

end Spec
