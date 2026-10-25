import Spec.L7.Context

/-!
# Layer 7: fit is sound up to `Unchecked` (`D-40`, `D-41`, `T-23`)
-/

namespace Spec

open Classical

/-- `D-41`. The collapse into a family `L`: a singleton shows as its member, a wider set inside `L`
as the whole of `L`, and any other wider set as `Unchecked`, the set of everything. -/
noncomputable def collapseTo {α : Type} (L X : Set α) : Set α :=
  if X.Subsingleton then X else if X ⊆ L then L else Set.univ

theorem subset_collapseTo {α : Type} (L X : Set α) : X ⊆ collapseTo L X := by
  unfold collapseTo
  split_ifs with hX hl
  · exact le_rfl
  · exact hl
  · exact Set.subset_univ X

theorem collapseTo_mono {α : Type} (L : Set α) {X Y : Set α} (h : X ⊆ Y) :
    collapseTo L X ⊆ collapseTo L Y := by
  by_cases hX : X.Subsingleton
  · simp only [collapseTo, hX, ite_true]
    exact h.trans (subset_collapseTo L Y)
  have hY : ¬ Y.Subsingleton := fun hY => hX (hY.anti h)
  by_cases hXl : X ⊆ L
  · simp only [collapseTo, hX, hY, hXl, ite_false, ite_true]
    split_ifs
    · exact le_rfl
    · exact Set.subset_univ _
  · have hYl : ¬ Y ⊆ L := fun hYl => hXl (h.trans hYl)
    simp only [collapseTo, hX, hY, hXl, hYl, ite_false]
    exact le_rfl

/-- `D-41`. The collapse: a singleton shows as its member, a wider set of lists as `List(Unchecked)`,
the set of every list, and any other wider set as `Unchecked`, the set of every data type. -/
noncomputable def collapse (X : Set DType) : Set DType :=
  collapseTo (Set.range DType.list) X

theorem collapse_of_subsingleton {X : Set DType} (hX : X.Subsingleton) : collapse X = X := by
  simp [collapse, collapseTo, hX]

theorem collapse_of_lists {X : Set DType} (hX : ¬ X.Subsingleton)
    (hl : X ⊆ Set.range DType.list) : collapse X = Set.range DType.list := by
  simp [collapse, collapseTo, hX, hl]

theorem collapse_of_wide {X : Set DType} (hX : ¬ X.Subsingleton)
    (hl : ¬ X ⊆ Set.range DType.list) : collapse X = Set.univ := by
  simp [collapse, collapseTo, hX, hl]

theorem subset_collapse (X : Set DType) : X ⊆ collapse X :=
  subset_collapseTo _ X

theorem collapse_mono {X Y : Set DType} (h : X ⊆ Y) : collapse X ⊆ collapse Y :=
  collapseTo_mono _ h

/-- The collapse keeps `Unchecked`. -/
theorem collapse_univ : collapse Set.univ = Set.univ := by
  apply collapse_of_wide
  · intro hs
    have := hs (Set.mem_univ (DType.scalar .int64)) (Set.mem_univ (DType.scalar .string))
    simp at this
  · intro hl
    obtain ⟨t, ht⟩ := hl (Set.mem_univ (DType.scalar .int64))
    cases ht

/-- The context rule read over a set of data types, then collapsed. -/
noncomputable def contextSet (k : Context) (ch : Character) (X : Set DType) : Set DType :=
  collapse (contextRule k ch '' X)

theorem contextSet_mono (k : Context) (ch : Character) {X Y : Set DType} (h : X ⊆ Y) :
    contextSet k ch X ⊆ contextSet k ch Y :=
  collapse_mono (Set.image_mono h)

/-- In the row context the context rule changes no data type, so it keeps `Unchecked`. -/
theorem contextSet_row_univ (ch : Character) : contextSet .row ch Set.univ = Set.univ := by
  have hid : contextRule .row ch = id := by
    funext t
    cases ch <;> rfl
  simp [contextSet, hid, collapse_univ]

/-- `D-39`. A declared slice, at the level `T-23` needs: the reads, the declared cells, the answer
set at each cell, the name and the character. -/
structure Slice where
  reads : Finset Name
  cells : Set Cell
  answer : Cell → Set DType
  name : Name
  character : Character

/-- The type of an argument, at the same level. -/
structure ArgType where
  needs : Finset Name
  answer : Cell → Set DType
  name : Name
  character : Character

/-- `D-40`. Two answers agree, or one of them is `Unchecked`. -/
def FitsAt (a d : Set DType) : Prop :=
  a = d ∨ a = Set.univ ∨ d = Set.univ

/-- `D-40`. Fit: the needs are within the reads, the names and characters are equal, and the answers
agree at every declared cell up to `Unchecked`. -/
def Fits (a : ArgType) (d : Slice) : Prop :=
  a.needs ⊆ d.reads ∧ a.name = d.name ∧ a.character = d.character ∧
    ∀ c ∈ d.cells, FitsAt (a.answer c) (d.answer c)

/-- Section 17.1 for an argument, at a frame whose declared names are `S` and whose data types fall
into the cell `c`. -/
noncomputable def ArgType.materialize (a : ArgType) (S : Finset Name) (k : Context) (c : Cell) :
    Option (Name × Set DType) :=
  if a.needs ⊆ S then some (a.name, contextSet k a.character (a.answer c)) else none

/-- Section 17.1 for a declared slice. -/
noncomputable def Slice.materialize (d : Slice) (S : Finset Name) (k : Context) (c : Cell) :
    Option (Name × Set DType) :=
  if d.reads ⊆ S then some (d.name, contextSet k d.character (d.answer c)) else none

/-- `T-23`. Wherever a declared slice materializes, an argument that fits it materializes too,
under the same name, and the data types agree up to `Unchecked`: they are equal, or one of the two
sets holds the other. -/
theorem T23.fit_sound (a : ArgType) (d : Slice) (h : Fits a d) (S : Finset Name) (k : Context)
    (c : Cell) (hc : c ∈ d.cells) (hS : d.reads ⊆ S) :
    ∃ X Y, a.materialize S k c = some (d.name, X) ∧ d.materialize S k c = some (d.name, Y) ∧
      (X = Y ∨ Y ⊆ X ∨ X ⊆ Y) := by
  obtain ⟨hneeds, hname, hchar, hcells⟩ := h
  refine ⟨contextSet k a.character (a.answer c), contextSet k d.character (d.answer c), ?_, ?_, ?_⟩
  · simp [ArgType.materialize, hneeds.trans hS, hname]
  · simp [Slice.materialize, hS]
  · rw [hchar]
    rcases hcells c hc with h | h | h
    · exact Or.inl (by rw [h])
    · refine Or.inr (Or.inl (contextSet_mono k d.character ?_))
      rw [h]
      exact Set.subset_univ _
    · refine Or.inr (Or.inr (contextSet_mono k d.character ?_))
      rw [h]
      exact Set.subset_univ _

/-- `T-23`. Where neither answer is `Unchecked`, the two data types are equal. -/
theorem T23.fit_sound_exact (a : ArgType) (d : Slice) (h : Fits a d) (S : Finset Name)
    (k : Context) (c : Cell) (hc : c ∈ d.cells) (hS : d.reads ⊆ S) (t u : DType)
    (ha : a.answer c = {t}) (hd : d.answer c = {u}) :
    a.materialize S k c = d.materialize S k c := by
  obtain ⟨hneeds, hname, hchar, hcells⟩ := h
  have hne : ∀ v : DType, ({v} : Set DType) ≠ Set.univ := by
    intro v hv
    have hmem : DType.list v ∈ ({v} : Set DType) := hv ▸ trivial
    have hsize := congrArg sizeOf (Set.mem_singleton_iff.mp hmem)
    simp at hsize
  have heq : a.answer c = d.answer c := by
    rcases hcells c hc with h | h | h
    · exact h
    · exact absurd (ha ▸ h) (hne t)
    · exact absurd (hd ▸ h) (hne u)
  simp [ArgType.materialize, Slice.materialize, hneeds.trans hS, hS, hname, hchar, heq]

/-- `T-23`. Fit alone does not give the same frame type: an argument whose answer is `Unchecked`
fits a declared `Int64`, and the two materialize differently. -/
theorem T23.unchecked_counterexample :
    ∃ (a : ArgType) (d : Slice), Fits a d ∧
      a.materialize ∅ .row (fun _ => .list) ≠ d.materialize ∅ .row (fun _ => .list) := by
  refine ⟨⟨∅, fun _ => Set.univ, [], .elementwise⟩,
    ⟨∅, Set.univ, fun _ => {.scalar .int64}, [], .elementwise⟩,
    ⟨subset_refl _, rfl, rfl, fun _ _ => Or.inr (Or.inl rfl)⟩, ?_⟩
  have hne : Set.univ ≠ contextSet .row .elementwise {.scalar .int64} := by
    intro h
    have hmem : DType.scalar .string ∈ contextSet .row .elementwise {.scalar .int64} :=
      h ▸ trivial
    simp [contextSet, collapse, collapseTo, contextRule] at hmem
  simp [ArgType.materialize, Slice.materialize, contextSet_row_univ, hne]

end Spec
