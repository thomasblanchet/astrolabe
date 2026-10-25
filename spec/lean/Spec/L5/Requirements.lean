import Spec.L2.Effects

/-!
# Layer 5: requirements and how the record decides them (`R-1` to `R-11`, `D-21`, `T-12`, `T-13`)
-/

namespace Spec

/-- `R-1` to `R-9`. A requirement that tests one column on its own. -/
inductive PerColumn
  | none
  | without (S : List Name)
  | only (S : List Name)
  | withoutPrefix (p : Name)
  | onlyPrefix (p : Name)
  | withoutSuffix (s : Name)
  | onlySuffix (s : Name)
  | allOf (F : DTSet)
  | noneOf (F : DTSet)

/-- `D-21`. The selector a per-column requirement tests each column with. -/
def PerColumn.pred : PerColumn → Selector
  | .none => .nameTest .always
  | .without S => .not (.nameTest (.oneOf S))
  | .only S => .nameTest (.oneOf S)
  | .withoutPrefix p => .not (.nameTest (.startsWith p))
  | .onlyPrefix p => .nameTest (.startsWith p)
  | .withoutSuffix s => .not (.nameTest (.endsWith s))
  | .onlySuffix s => .nameTest (.endsWith s)
  | .allOf F => .ofType F
  | .noneOf F => .not (.ofType F)

/-- A per-column requirement holds of a column set when every column passes its predicate. -/
def PerColumn.holds (ρ : PerColumn) (X : ColumnSet) : Prop :=
  ∀ c ∈ X, c ∈ ρ.pred.denote

/-- The columns a record allows: a name the absence rule does not hold of, and a data type in the
type set. Every column set a record describes is a subset of these. -/
def Record.allowed (r : Record) : Set (Name × DType) :=
  {c | c.1 ∉ r.absent.denote ∧ c.2 ∈ r.types.denote}

/-- `T-12`. The condition each per-column row of the table states on the record. -/
def PerColumn.decides (ρ : PerColumn) (r : Record) : Prop :=
  match ρ with
  | .none => True
  | .without S => ∀ n ∈ S, n ∈ r.absent.denote
  | .only S => ∀ n, n ∉ S → n ∈ r.absent.denote
  | .withoutPrefix p => ∀ n, p <+: n → n ∈ r.absent.denote
  | .onlyPrefix p => ∀ n, ¬ p <+: n → n ∈ r.absent.denote
  | .withoutSuffix s => ∀ n, s <:+ n → n ∈ r.absent.denote
  | .onlySuffix s => ∀ n, ¬ s <:+ n → n ∈ r.absent.denote
  | .allOf F => r.types.denote ⊆ F.denote
  | .noneOf F => r.types.denote ∩ F.denote = ∅

/-- `T-12`, the exact form. A per-column requirement holds of every column set a record describes
exactly when every column the record allows passes it. -/
theorem T12.holds_iff_allowed (ρ : PerColumn) (r : Record) :
    (∀ X, describes r X → ρ.holds X) ↔ r.allowed ⊆ ρ.pred.denote := by
  constructor
  · intro h c hc
    have hX : describes r {c} := by
      rintro n t hnt
      rw [Set.mem_singleton_iff] at hnt
      subst hnt
      exact hc
    exact h {c} hX c rfl
  · intro h X hX c hc
    exact h (hX c.1 c.2 hc)

/-- `T-12`, the table. A per-column requirement holds of every column set a record describes
exactly when its row holds, or the record allows no column: an empty type set, or an absence rule
that holds of every name. -/
theorem T12.decide_iff (ρ : PerColumn) (r : Record) :
    (∀ X, describes r X → ρ.holds X) ↔
      ρ.decides r ∨ r.types.denote = ∅ ∨ r.absent.denote = Set.univ := by
  rw [T12.holds_iff_allowed]
  constructor
  · intro h
    by_cases hΘ : r.types.denote = ∅
    · exact Or.inr (Or.inl hΘ)
    by_cases hP : r.absent.denote = Set.univ
    · exact Or.inr (Or.inr hP)
    refine Or.inl ?_
    obtain ⟨t, ht⟩ := Set.nonempty_iff_ne_empty.mpr hΘ
    obtain ⟨m, hm⟩ : ∃ m, m ∉ r.absent.denote := by
      by_contra hno
      push Not at hno
      exact hP (Set.eq_univ_of_forall hno)
    cases ρ with
    | none => trivial
    | allOf F => intro u hu; simpa [PerColumn.pred, Selector.denote] using h (show (m, u) ∈ r.allowed from ⟨hm, hu⟩)
    | noneOf F =>
      ext u
      simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false, not_and]
      intro hu huF
      simpa [PerColumn.pred, Selector.denote, huF] using h (show (m, u) ∈ r.allowed from ⟨hm, hu⟩)
    | without S =>
      intro n hn; by_contra hnP
      simpa [PerColumn.pred, Selector.denote, Absence.denote, hn] using h (show (n, t) ∈ r.allowed from ⟨hnP, ht⟩)
    | only S =>
      intro n hn; by_contra hnP
      simpa [PerColumn.pred, Selector.denote, Absence.denote, hn] using h (show (n, t) ∈ r.allowed from ⟨hnP, ht⟩)
    | withoutPrefix p =>
      intro n hn; by_contra hnP
      simpa [PerColumn.pred, Selector.denote, Absence.denote, hn] using h (show (n, t) ∈ r.allowed from ⟨hnP, ht⟩)
    | onlyPrefix p =>
      intro n hn; by_contra hnP
      simpa [PerColumn.pred, Selector.denote, Absence.denote, hn] using h (show (n, t) ∈ r.allowed from ⟨hnP, ht⟩)
    | withoutSuffix s =>
      intro n hn; by_contra hnP
      simpa [PerColumn.pred, Selector.denote, Absence.denote, hn] using h (show (n, t) ∈ r.allowed from ⟨hnP, ht⟩)
    | onlySuffix s =>
      intro n hn; by_contra hnP
      simpa [PerColumn.pred, Selector.denote, Absence.denote, hn] using h (show (n, t) ∈ r.allowed from ⟨hnP, ht⟩)
  · rintro (h | hΘ | hP) ⟨n, t⟩ ⟨hn, ht⟩
    · cases ρ with
      | none => simp [PerColumn.pred, Selector.denote, Absence.denote]
      | allOf F => simpa [PerColumn.pred, Selector.denote] using h ht
      | noneOf F =>
        simp only [PerColumn.pred, Selector.denote, Set.mem_compl_iff, Set.mem_ofPred_eq]
        intro htF
        have : t ∈ r.types.denote ∩ F.denote := ⟨ht, htF⟩
        simp [PerColumn.decides] at h
        rw [h] at this
        exact this
      | without S =>
        simp only [PerColumn.pred, Selector.denote, Absence.denote, Set.mem_compl_iff,
          Set.mem_ofPred_eq]
        exact fun hS => hn (h n hS)
      | only S =>
        simp only [PerColumn.pred, Selector.denote, Absence.denote, Set.mem_ofPred_eq]
        by_contra hS; exact hn (h n hS)
      | withoutPrefix p =>
        simp only [PerColumn.pred, Selector.denote, Absence.denote, Set.mem_compl_iff,
          Set.mem_ofPred_eq]
        exact fun hp => hn (h n hp)
      | onlyPrefix p =>
        simp only [PerColumn.pred, Selector.denote, Absence.denote, Set.mem_ofPred_eq]
        by_contra hp; exact hn (h n hp)
      | withoutSuffix s =>
        simp only [PerColumn.pred, Selector.denote, Absence.denote, Set.mem_compl_iff,
          Set.mem_ofPred_eq]
        exact fun hs => hn (h n hs)
      | onlySuffix s =>
        simp only [PerColumn.pred, Selector.denote, Absence.denote, Set.mem_ofPred_eq]
        by_contra hs; exact hn (h n hs)
    · rw [hΘ] at ht; exact ht.elim
    · exact absurd (hP ▸ Set.mem_univ n) hn

/-- The names of a column set. -/
def names (X : ColumnSet) : Set Name :=
  {n | ∃ t, (n, t) ∈ X}

/-- `R-10`, the exact form. Two remainders bound independently share no name under every binding
exactly when no name is allowed by both absence rules, or one record allows no column. -/
theorem T12.disjoint_iff (a b : Record) :
    (∀ X Y, describes a X → describes b Y → names X ∩ names Y = ∅) ↔
      (a.absent.denoteᶜ ∩ b.absent.denoteᶜ = ∅) ∨ a.allowed = ∅ ∨ b.allowed = ∅ := by
  constructor
  · intro h
    by_cases ha : a.allowed = ∅
    · exact Or.inr (Or.inl ha)
    by_cases hb : b.allowed = ∅
    · exact Or.inr (Or.inr hb)
    refine Or.inl ?_
    obtain ⟨⟨_, ta⟩, _, hta⟩ := Set.nonempty_iff_ne_empty.mpr ha
    obtain ⟨⟨_, tb⟩, _, htb⟩ := Set.nonempty_iff_ne_empty.mpr hb
    ext n
    simp only [Set.mem_inter_iff, Set.mem_compl_iff, Set.mem_empty_iff_false, iff_false, not_and]
    intro hna hnb
    have hX : describes a {(n, ta)} := by
      rintro m u hmu; rw [Set.mem_singleton_iff, Prod.mk.injEq] at hmu
      obtain ⟨rfl, rfl⟩ := hmu; exact ⟨hna, hta⟩
    have hY : describes b {(n, tb)} := by
      rintro m u hmu; rw [Set.mem_singleton_iff, Prod.mk.injEq] at hmu
      obtain ⟨rfl, rfl⟩ := hmu; exact ⟨hnb, htb⟩
    have := h _ _ hX hY
    have hn : n ∈ names {(n, ta)} ∩ names {(n, tb)} := ⟨⟨ta, rfl⟩, ⟨tb, rfl⟩⟩
    rw [this] at hn
    exact hn
  · rintro (h | ha | hb) X Y hX hY
    · ext n
      simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false, not_and, names,
        Set.mem_ofPred_eq]
      rintro ⟨t, ht⟩ ⟨u, hu⟩
      have : n ∈ a.absent.denoteᶜ ∩ b.absent.denoteᶜ := ⟨(hX n t ht).1, (hY n u hu).1⟩
      rw [h] at this
      exact this
    · ext n
      simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false, not_and, names,
        Set.mem_ofPred_eq]
      rintro ⟨t, ht⟩
      have : (n, t) ∈ a.allowed := hX n t ht
      rw [ha] at this
      exact this.elim
    · ext n
      simp only [Set.mem_inter_iff, Set.mem_empty_iff_false, iff_false, not_and, names,
        Set.mem_ofPred_eq]
      rintro _ ⟨u, hu⟩
      have : (n, u) ∈ b.allowed := hY n u hu
      rw [hb] at this
      exact this.elim

/-- `R-11`, the row the table used to state: every name the checked remainder can hold, the named
one can hold too. It is not sound. -/
def withinRow (a b : Record) : Prop :=
  a.absent.denoteᶜ ⊆ b.absent.denoteᶜ

/-- `R-11`. The old row is unsound: two unconstrained remainders pass it, and a caller whose first
remainder holds a column while the second is empty breaks `within=`. -/
theorem T12.within_row_unsound :
    ∃ a b : Record, withinRow a b ∧
      ¬ ∀ X Y, describes a X → describes b Y → names X ⊆ names Y := by
  refine ⟨⟨.never, .univ⟩, ⟨.never, .univ⟩, fun _ h => h, fun h => ?_⟩
  have hX : describes ⟨.never, .univ⟩ {(['x'], .scalar .int64)} := by
    rintro n t hnt
    rw [Set.mem_singleton_iff, Prod.mk.injEq] at hnt
    obtain ⟨rfl, rfl⟩ := hnt
    exact ⟨by simp [Absence.denote], trivial⟩
  have hY : describes ⟨.never, .univ⟩ ∅ := fun _ _ h => h.elim
  obtain ⟨t, ht⟩ := h _ _ hX hY ⟨.scalar .int64, rfl⟩
  exact ht

/-- `R-11`, the exact form. From the two records alone, `within=` holds under every binding exactly
when the checked record allows no column. -/
theorem T12.within_iff (a b : Record) :
    (∀ X Y, describes a X → describes b Y → names X ⊆ names Y) ↔ a.allowed = ∅ := by
  constructor
  · intro h
    ext ⟨n, t⟩
    simp only [Set.mem_empty_iff_false, iff_false]
    intro hc
    have hX : describes a {(n, t)} := by
      rintro m u hmu; rw [Set.mem_singleton_iff, Prod.mk.injEq] at hmu
      obtain ⟨rfl, rfl⟩ := hmu; exact hc
    have hY : describes b ∅ := fun _ _ h => h.elim
    obtain ⟨u, hu⟩ := h _ _ hX hY ⟨t, rfl⟩
    exact hu
  · rintro ha X Y hX _ n ⟨t, ht⟩
    have : (n, t) ∈ a.allowed := hX n t ht
    rw [ha] at this
    exact this.elim

/-- `T-13`. Disjointness from a union is disjointness from each part. -/
theorem T13.disjoint_union_iff (R A B : Set Name) :
    R ∩ (A ∪ B) = ∅ ↔ R ∩ A = ∅ ∧ R ∩ B = ∅ := by
  rw [Set.inter_union_distrib_left, Set.union_empty_iff]

/-- `T-13`. Containment in one part gives containment in the union. -/
theorem T13.subset_union_of_subset (R A B : Set Name) (h : R ⊆ A ∨ R ⊆ B) : R ⊆ A ∪ B := by
  rcases h with h | h
  · exact h.trans Set.subset_union_left
  · exact h.trans Set.subset_union_right

/-- `T-13`. Containment in the union does not give containment in either part. -/
theorem T13.subset_union_not_imp :
    ∃ R A B : Set Name, R ⊆ A ∪ B ∧ ¬ R ⊆ A ∧ ¬ R ⊆ B := by
  refine ⟨{['a'], ['b']}, {['a']}, {['b']}, ?_, ?_, ?_⟩
  · intro n hn
    simp only [Set.mem_insert_iff, Set.mem_singleton_iff] at hn
    rcases hn with rfl | rfl <;> simp
  · intro h
    have := h (Set.mem_insert_of_mem _ rfl)
    simp at this
  · intro h
    have := h (Set.mem_insert _ _)
    simp at this

end Spec
