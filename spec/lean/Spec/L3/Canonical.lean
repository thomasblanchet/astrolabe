import Spec.L2.Effects

/-!
# Layer 3: the canonical form (`D-22`, `D-23`, `T-4`, `T-5`, `T-8`)

A canonical form is a base, then an optional filter, a suffix, a prefix and an optional retype.
`normalize` computes it by structural induction on the expression.
-/

namespace Spec

mutual
/-- `D-22`. A canonical form: a base, then a filter, a suffix, a prefix and a retype. -/
inductive Canon (V : Type) where
  | mk (base : Base V) (filter : Option Selector) (suffix pre : Name) (retype : Option DType)

/-- `D-22`. A base: a variable, a union of canonical forms, or nothing. -/
inductive Base (V : Type) where
  | var (v : V)
  | nothing
  | union (parts : List (Canon V))
end

variable {V : Type}

/-- The column a base column becomes under a form's suffix, prefix and retype. -/
def renameColumn (suffix pre : Name) (retype : Option DType) (c : Name × DType) : Name × DType :=
  (pre ++ (c.1 ++ suffix), retype.getD c.2)

/-- The columns a filter slot keeps. -/
def filterSet : Option Selector → Set (Name × DType)
  | none => Set.univ
  | some F => F.denote

mutual
/-- `D-22`. The column set a canonical form denotes under a binding. -/
def Canon.denote (β : V → ColumnSet) : Canon V → ColumnSet
  | .mk b f s p r => renameColumn s p r '' (b.denote β ∩ filterSet f)

/-- The column set a base denotes under a binding. -/
def Base.denote (β : V → ColumnSet) : Base V → ColumnSet
  | .var v => β v
  | .nothing => ∅
  | .union ps => Canon.denoteList β ps

/-- The union of the column sets a list of forms denotes. -/
def Canon.denoteList (β : V → ColumnSet) : List (Canon V) → ColumnSet
  | [] => ∅
  | k :: ks => k.denote β ∪ Canon.denoteList β ks
end

/-- The form `E-2` produces: nothing, with every slot empty. -/
def nothingForm : Canon V :=
  .mk .nothing none [] [] none

/-- A form denotes nothing when its base is nothing. -/
def Canon.isNothing : Canon V → Bool
  | .mk .nothing _ _ _ _ => true
  | _ => false

/-- The operands a form adds to a union: its parts if it is a bare union, else itself. -/
def Canon.flatten : Canon V → List (Canon V)
  | .mk (.union ps) none [] [] none => ps
  | k => [k]

/-- What `normalize` needs beyond the rules of `D-22`, each with the property that makes it
correct: the name translation of `D-23`, the two tests that settle a filter, and the arrangement
of a union's operands (the sort and the removal of repeats). -/
structure Normalizer (V : Type) where
  /-- `D-23`. A rule that holds of `n` exactly when the given rule holds of `pre ++ n ++ suffix`. -/
  translateName : Name → Name → Absence → Absence
  translateName_spec : ∀ pre suffix P n,
    n ∈ (translateName pre suffix P).denote ↔ pre ++ (n ++ suffix) ∈ P.denote
  /-- A sound test that a selector picks no column. -/
  picksNone : Selector → Bool
  picksNone_spec : ∀ C, picksNone C = true → C.denote = ∅
  /-- A sound test that a selector picks every column. -/
  picksAll : Selector → Bool
  picksAll_spec : ∀ C, picksAll C = true → C.denote = Set.univ
  /-- The arrangement of a union's operands: a sort that can drop repeats. -/
  arrange : List (Canon V) → List (Canon V)
  arrange_spec : ∀ l k, k ∈ arrange l ↔ k ∈ l

open Classical in
/-- `D-23`. The selector that tests a base column the way `C` tests the column it becomes under a
form's retype, prefix and suffix. -/
noncomputable def translate (O : Normalizer V) (suffix pre : Name) (retype : Option DType) :
    Selector → Selector
  | .nameTest P => .nameTest (O.translateName pre suffix P)
  | .ofType F =>
    match retype with
    | some U => .nameTest (if U ∈ F.denote then .always else .never)
    | none => .ofType F
  | .not C => .not (translate O suffix pre retype C)
  | .and C D => .and (translate O suffix pre retype C) (translate O suffix pre retype D)
  | .or C D => .or (translate O suffix pre retype C) (translate O suffix pre retype D)

/-- `D-22`. The union of a list of forms, with empty operands dropped and a single operand kept
bare. -/
def mkUnion (O : Normalizer V) (ps : List (Canon V)) : Canon V :=
  match O.arrange (ps.filter fun k => !k.isNothing) with
  | [] => nothingForm
  | [k] => k
  | qs => .mk (.union qs) none [] [] none

/-- `D-22`. Add a filter to a form. A filter that picks nothing gives the empty form, and one that
picks everything is dropped. -/
noncomputable def withFilter (O : Normalizer V) (C : Selector) : Canon V → Canon V
  | .mk b f s p r =>
    let T := translate O s p r C
    let F := match f with
      | none => T
      | some F₀ => .and F₀ T
    if O.picksNone F then nothingForm
    else if O.picksAll F then .mk b none s p r
    else .mk b (some F) s p r

/-- Set the retype slot. -/
def Canon.withRetype (U : DType) : Canon V → Canon V
  | .mk b f s p _ => .mk b f s p (some U)

/-- Append to the suffix slot. -/
def Canon.withSuffix (x : Name) : Canon V → Canon V
  | .mk b f s p r => .mk b f (s ++ x) p r

/-- Prepend to the prefix slot. -/
def Canon.withPrefix (x : Name) : Canon V → Canon V
  | .mk b f s p r => .mk b f s (x ++ p) r

/-- `D-22`. Apply one step of an effect to a form. Every effect acts on each column separately, so
over a bare union the step applies to each part and the parts unite again: a union form carries no
effect. -/
def Canon.distribute (O : Normalizer V) (g : Canon V → Canon V) : Canon V → Canon V
  | .mk (.union ps) none [] [] none => mkUnion O (ps.map g)
  | k => g k

/-- `D-22`. The canonical form of an expression, by structural induction. -/
noncomputable def normalize (O : Normalizer V) : RemExpr V → Canon V
  | .var v => .mk (.var v) none [] [] none
  | .both e₁ e₂ => mkUnion O ((normalize O e₁).flatten ++ (normalize O e₂).flatten)
  | .app .unchanged e => normalize O e
  | .app .emptied _ => nothingForm
  | .app (.retyped U) e => (normalize O e).distribute O (Canon.withRetype U)
  | .app (.suffixed x) e => (normalize O e).distribute O (Canon.withSuffix x)
  | .app (.prefixed x) e => (normalize O e).distribute O (Canon.withPrefix x)
  | .app (.filtered C) e => (normalize O e).distribute O (withFilter O C)

/-- `D-23`. A translated selector picks a base column exactly when the original picks the column
that base column becomes. -/
theorem D23.translate_correct (O : Normalizer V) (suffix pre : Name) (retype : Option DType)
    (C : Selector) (c : Name × DType) :
    c ∈ (translate O suffix pre retype C).denote ↔
      renameColumn suffix pre retype c ∈ C.denote := by
  induction C with
  | nameTest P =>
    simp [translate, Selector.denote, renameColumn, O.translateName_spec]
  | ofType F =>
    cases retype with
    | none => simp [translate, Selector.denote, renameColumn]
    | some U =>
      by_cases h : U ∈ F.denote
      · simp [translate, Selector.denote, renameColumn, h, Absence.denote]
      · simp [translate, Selector.denote, renameColumn, h, Absence.denote]
  | not C ih => simp [translate, Selector.denote, ih]
  | and C D ihC ihD => simp [translate, Selector.denote, ihC, ihD]
  | or C D ihC ihD => simp [translate, Selector.denote, ihC, ihD]

/-- A list of forms denotes a column exactly when one of its forms does. -/
theorem Canon.mem_denoteList (β : V → ColumnSet) (ps : List (Canon V)) (c : Name × DType) :
    c ∈ Canon.denoteList β ps ↔ ∃ k ∈ ps, c ∈ k.denote β := by
  induction ps with
  | nil => simp [Canon.denoteList]
  | cons k ks ih => simp [Canon.denoteList, ih]

/-- A form with nothing as its base denotes nothing. -/
theorem Canon.denote_of_isNothing (β : V → ColumnSet) (k : Canon V) (h : k.isNothing = true) :
    k.denote β = ∅ := by
  match k, h with
  | .mk .nothing _ _ _ _, _ => simp [Canon.denote, Base.denote]

/-- The empty form denotes nothing. -/
theorem nothingForm_denote (β : V → ColumnSet) : (nothingForm : Canon V).denote β = ∅ :=
  Canon.denote_of_isNothing β _ rfl

/-- Renaming with empty slots is the identity. -/
theorem renameColumn_trivial (c : Name × DType) : renameColumn [] [] none c = c := by
  simp [renameColumn]

/-- A form with every slot empty denotes what its base denotes. -/
theorem Canon.denote_trivial (β : V → ColumnSet) (b : Base V) :
    (Canon.mk b none [] [] none).denote β = b.denote β := by
  ext c
  simp [Canon.denote, filterSet, renameColumn_trivial]

/-- Flattening keeps what a form denotes. -/
theorem Canon.denote_flatten (β : V → ColumnSet) (k : Canon V) :
    Canon.denoteList β k.flatten = k.denote β := by
  match k with
  | .mk (.union ps) none [] [] none =>
    simp only [Canon.flatten]
    rw [Canon.denote_trivial]
    rfl
  | .mk (.var _) _ _ _ _ | .mk .nothing _ _ _ _
  | .mk (.union _) (some _) _ _ _ | .mk (.union _) none (_ :: _) _ _
  | .mk (.union _) none [] (_ :: _) _ | .mk (.union _) none [] [] (some _) =>
    simp [Canon.flatten, Canon.denoteList]

/-- The union `mkUnion` builds denotes the union of its operands. -/
theorem mkUnion_denote (O : Normalizer V) (β : V → ColumnSet) (ps : List (Canon V)) :
    (mkUnion O ps).denote β = Canon.denoteList β ps := by
  have key : ∀ c, c ∈ Canon.denoteList β (O.arrange (ps.filter fun k => !k.isNothing)) ↔
      c ∈ Canon.denoteList β ps := by
    intro c
    simp only [Canon.mem_denoteList, O.arrange_spec, List.mem_filter]
    constructor
    · rintro ⟨k, ⟨hk, _⟩, hc⟩
      exact ⟨k, hk, hc⟩
    · rintro ⟨k, hk, hc⟩
      refine ⟨k, ⟨hk, ?_⟩, hc⟩
      cases h : k.isNothing
      · rfl
      · rw [Canon.denote_of_isNothing β k h] at hc
        exact absurd hc (Set.notMem_empty c)
  ext c
  rw [← key]
  unfold mkUnion
  generalize O.arrange (ps.filter fun k => !k.isNothing) = L
  match L with
  | [] => simp [Canon.denoteList, nothingForm_denote]
  | [k] => simp [Canon.denoteList]
  | k₁ :: k₂ :: ks => rw [Canon.denote_trivial]; rfl

/-- One filter step: a filter `F` that keeps exactly the base columns the old filter keeps and
whose renamed column `C` picks gives the form's columns that `C` picks. -/
theorem filter_step (O : Normalizer V) (β : V → ColumnSet) (C : Selector) (b : Base V)
    (f : Option Selector) (s p : Name) (r : Option DType) (F : Selector)
    (hF : ∀ y, y ∈ F.denote ↔ y ∈ filterSet f ∧ renameColumn s p r y ∈ C.denote) :
    (if O.picksNone F then nothingForm
      else if O.picksAll F then Canon.mk b none s p r else Canon.mk b (some F) s p r).denote β =
      (Canon.mk b f s p r).denote β ∩ C.denote := by
  ext c
  split_ifs with hn ha
  · rw [nothingForm_denote]
    simp only [Set.mem_empty_iff_false, false_iff, Set.mem_inter_iff, Canon.denote,
      Set.mem_image, not_and]
    rintro ⟨y, ⟨_, hf⟩, rfl⟩ hc
    have hy := (hF y).2 ⟨hf, hc⟩
    rw [O.picksNone_spec _ hn] at hy
    exact hy
  · simp only [Canon.denote, Set.mem_inter_iff, Set.mem_image, filterSet, Set.mem_univ,
      and_true]
    constructor
    · rintro ⟨y, hb, rfl⟩
      have hy := (hF y).1 (by rw [O.picksAll_spec _ ha]; trivial)
      exact ⟨⟨y, ⟨hb, hy.1⟩, rfl⟩, hy.2⟩
    · rintro ⟨⟨y, ⟨hb, _⟩, rfl⟩, _⟩
      exact ⟨y, hb, rfl⟩
  · simp only [Canon.denote, Set.mem_inter_iff, Set.mem_image, filterSet]
    constructor
    · rintro ⟨y, ⟨hb, hy⟩, rfl⟩
      obtain ⟨hf, hc⟩ := (hF y).1 hy
      exact ⟨⟨y, ⟨hb, hf⟩, rfl⟩, hc⟩
    · rintro ⟨⟨y, ⟨hb, hf⟩, rfl⟩, hc⟩
      exact ⟨y, ⟨hb, (hF y).2 ⟨hf, hc⟩⟩, rfl⟩

/-- `D-22`. Adding a filter keeps the form's columns that `C` picks. -/
theorem withFilter_denote (O : Normalizer V) (β : V → ColumnSet) (C : Selector) (k : Canon V) :
    (withFilter O C k).denote β = k.denote β ∩ C.denote := by
  match k with
  | .mk b none s p r =>
    exact filter_step O β C b none s p r _ fun y => by
      simp [filterSet, D23.translate_correct]
  | .mk b (some F₀) s p r =>
    exact filter_step O β C b (some F₀) s p r _ fun y => by
      simp [filterSet, Selector.denote, D23.translate_correct]

/-- Every effect distributes over a union of column sets. -/
theorem Effect.apply_union (ε : Effect) (X Y : ColumnSet) :
    ε.apply (X ∪ Y) = ε.apply X ∪ ε.apply Y := by
  cases ε with
  | unchanged => rfl
  | emptied => simp [Effect.apply]
  | filtered C => exact Set.union_inter_distrib_right X Y C.denote
  | suffixed s =>
    ext c
    simp only [Effect.apply, Set.mem_union, Set.mem_ofPred_eq]
    constructor
    · rintro ⟨n, t, h | h, rfl⟩
      · exact Or.inl ⟨n, t, h, rfl⟩
      · exact Or.inr ⟨n, t, h, rfl⟩
    · rintro (⟨n, t, h, rfl⟩ | ⟨n, t, h, rfl⟩)
      · exact ⟨n, t, Or.inl h, rfl⟩
      · exact ⟨n, t, Or.inr h, rfl⟩
  | prefixed p =>
    ext c
    simp only [Effect.apply, Set.mem_union, Set.mem_ofPred_eq]
    constructor
    · rintro ⟨n, t, h | h, rfl⟩
      · exact Or.inl ⟨n, t, h, rfl⟩
      · exact Or.inr ⟨n, t, h, rfl⟩
    · rintro (⟨n, t, h, rfl⟩ | ⟨n, t, h, rfl⟩)
      · exact ⟨n, t, Or.inl h, rfl⟩
      · exact ⟨n, t, Or.inr h, rfl⟩
  | retyped U =>
    ext c
    simp only [Effect.apply, Set.mem_union, Set.mem_ofPred_eq]
    constructor
    · rintro ⟨n, t, h | h, rfl⟩
      · exact Or.inl ⟨n, t, h, rfl⟩
      · exact Or.inr ⟨n, t, h, rfl⟩
    · rintro (⟨n, t, h, rfl⟩ | ⟨n, t, h, rfl⟩)
      · exact ⟨n, t, Or.inl h, rfl⟩
      · exact ⟨n, t, Or.inr h, rfl⟩

/-- Every effect keeps the empty column set empty. -/
theorem Effect.apply_empty (ε : Effect) : ε.apply ∅ = ∅ := by
  cases ε <;> simp [Effect.apply]

/-- A step that does what an effect does to every form does it to each part of a list. -/
theorem denoteList_map (β : V → ColumnSet) (ε : Effect) (g : Canon V → Canon V)
    (hg : ∀ k, (g k).denote β = ε.apply (k.denote β)) (ps : List (Canon V)) :
    Canon.denoteList β (ps.map g) = ε.apply (Canon.denoteList β ps) := by
  induction ps with
  | nil => simp [Canon.denoteList, Effect.apply_empty]
  | cons k ks ih => simp [Canon.denoteList, hg, ih, Effect.apply_union]

/-- `D-22`. Distributing a step over a union does what the effect does to the whole union. -/
theorem D22.distribute_denote (O : Normalizer V) (β : V → ColumnSet) (ε : Effect)
    (g : Canon V → Canon V) (hg : ∀ k, (g k).denote β = ε.apply (k.denote β)) (k : Canon V) :
    (k.distribute O g).denote β = ε.apply (k.denote β) := by
  match k with
  | .mk (.union ps) none [] [] none =>
    simp only [Canon.distribute]
    rw [mkUnion_denote, denoteList_map β ε g hg, Canon.denote_trivial]
    rfl
  | .mk (.var _) _ _ _ _ | .mk .nothing _ _ _ _
  | .mk (.union _) (some _) _ _ _ | .mk (.union _) none (_ :: _) _ _
  | .mk (.union _) none [] (_ :: _) _ | .mk (.union _) none [] [] (some _) =>
    simp only [Canon.distribute]
    exact hg _

/-- Setting the retype slot does what `E-6` does. -/
theorem withRetype_denote (β : V → ColumnSet) (U : DType) (k : Canon V) :
    (k.withRetype U).denote β = (Effect.retyped U).apply (k.denote β) := by
  match k with
  | .mk b f s p r =>
    ext c
    simp only [Canon.withRetype, Canon.denote, Effect.apply, Set.mem_image,
      Set.mem_ofPred_eq, renameColumn, Option.getD_some]
    constructor
    · rintro ⟨x, hx, rfl⟩
      exact ⟨_, _, ⟨x, hx, rfl⟩, rfl⟩
    · rintro ⟨n, t, ⟨x, hx, hxe⟩, rfl⟩
      simp only [Prod.mk.injEq] at hxe
      exact ⟨x, hx, by rw [hxe.1]⟩

/-- Appending to the suffix slot does what `E-5a` does. -/
theorem withSuffix_denote (β : V → ColumnSet) (x : Name) (k : Canon V) :
    (k.withSuffix x).denote β = (Effect.suffixed x).apply (k.denote β) := by
  match k with
  | .mk b f s p r =>
    ext c
    simp only [Canon.withSuffix, Canon.denote, Effect.apply, Set.mem_image,
      Set.mem_ofPred_eq, renameColumn]
    constructor
    · rintro ⟨y, hy, rfl⟩
      exact ⟨_, _, ⟨y, hy, rfl⟩, by simp⟩
    · rintro ⟨n, t, ⟨y, hy, hye⟩, rfl⟩
      simp only [Prod.mk.injEq] at hye
      exact ⟨y, hy, by rw [← hye.1, ← hye.2]; simp⟩

/-- Prepending to the prefix slot does what `E-5b` does. -/
theorem withPrefix_denote (β : V → ColumnSet) (x : Name) (k : Canon V) :
    (k.withPrefix x).denote β = (Effect.prefixed x).apply (k.denote β) := by
  match k with
  | .mk b f s p r =>
    ext c
    simp only [Canon.withPrefix, Canon.denote, Effect.apply, Set.mem_image,
      Set.mem_ofPred_eq, renameColumn]
    constructor
    · rintro ⟨y, hy, rfl⟩
      exact ⟨_, _, ⟨y, hy, rfl⟩, by simp⟩
    · rintro ⟨n, t, ⟨y, hy, hye⟩, rfl⟩
      simp only [Prod.mk.injEq] at hye
      exact ⟨y, hy, by rw [← hye.1, ← hye.2]; simp⟩

/-- `T-5`. The canonical form denotes what the expression denotes, under every binding. -/
theorem T5.normalize_preserves_denotation (O : Normalizer V) (β : V → ColumnSet)
    (e : RemExpr V) : (normalize O e).denote β = e.denote β := by
  induction e with
  | var v => exact Canon.denote_trivial β (.var v)
  | both e₁ e₂ ih₁ ih₂ =>
    rw [normalize, mkUnion_denote]
    ext c
    simp only [Canon.mem_denoteList, List.mem_append, RemExpr.denote, Set.mem_union]
    rw [← ih₁, ← ih₂, ← Canon.denote_flatten β (normalize O e₁),
      ← Canon.denote_flatten β (normalize O e₂)]
    simp only [Canon.mem_denoteList]
    constructor
    · rintro ⟨k, hk | hk, hc⟩
      · exact Or.inl ⟨k, hk, hc⟩
      · exact Or.inr ⟨k, hk, hc⟩
    · rintro (⟨k, hk, hc⟩ | ⟨k, hk, hc⟩)
      · exact ⟨k, Or.inl hk, hc⟩
      · exact ⟨k, Or.inr hk, hc⟩
  | app ε e ih =>
    cases ε with
    | unchanged => exact ih
    | emptied => simp [normalize, nothingForm_denote, RemExpr.denote, Effect.apply]
    | retyped U =>
      rw [normalize, RemExpr.denote, ← ih]
      exact D22.distribute_denote O β _ _ (withRetype_denote β U) _
    | suffixed x =>
      rw [normalize, RemExpr.denote, ← ih]
      exact D22.distribute_denote O β _ _ (withSuffix_denote β x) _
    | prefixed x =>
      rw [normalize, RemExpr.denote, ← ih]
      exact D22.distribute_denote O β _ _ (withPrefix_denote β x) _
    | filtered C =>
      rw [normalize, RemExpr.denote, ← ih]
      exact D22.distribute_denote O β _ _ (withFilter_denote O β C) _

/-- `T-4`. `normalize` is a total function: every expression has exactly one canonical form. -/
theorem T4.canonical_form_unique (O : Normalizer V) (e : RemExpr V) :
    ∃! k : Canon V, k = normalize O e :=
  ⟨normalize O e, rfl, fun _ h => h⟩

/-- `T-8`. Two expressions with the same canonical form denote the same column set under every
binding. -/
theorem T8.equal_forms_equal_denotations (O : Normalizer V) (e₁ e₂ : RemExpr V)
    (h : normalize O e₁ = normalize O e₂) (β : V → ColumnSet) :
    e₁.denote β = e₂.denote β := by
  rw [← T5.normalize_preserves_denotation O β e₁, ← T5.normalize_preserves_denotation O β e₂, h]

end Spec
