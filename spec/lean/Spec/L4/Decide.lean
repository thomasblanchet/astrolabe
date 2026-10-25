import Spec.L4.Regular
import Spec.L1.Selector

/-!
# Layer 4: containment between two selectors is decidable (`T-9`)

Two selectors mention finitely many data-type sets. Every data type behaves, under both selectors,
like one of finitely many witnesses. At one data type, the names a selector picks form a regular
language, and containment between two regular languages is settled by the words shorter than the
state count of one automaton. So containment reduces to finitely many checks, each over a finite
set of words: the characters are finite too.
-/

namespace Spec

open Language Classical Computability

/-- The data-type sets a selector mentions. -/
def Selector.dtsets : Selector → List DTSet
  | .nameTest _ => []
  | .ofType F => [F]
  | .not C => C.dtsets
  | .and C D => C.dtsets ++ D.dtsets
  | .or C D => C.dtsets ++ D.dtsets

/-- `D-11`. Every `matching` in a selector's name tests decides a regular set. -/
def Selector.MatchesRegular : Selector → Prop
  | .nameTest P => P.MatchesRegular
  | .ofType _ => True
  | .not C => C.MatchesRegular
  | .and C D => C.MatchesRegular ∧ D.MatchesRegular
  | .or C D => C.MatchesRegular ∧ D.MatchesRegular

/-- The names a selector picks at one data type. -/
def nameLang (C : Selector) (t : DType) : Set Name :=
  {n | (n, t) ∈ C.denote}

/-- Two data types that every data-type set of a selector treats alike are picked with the same
names. -/
theorem T9.same_types (C : Selector) (t w : DType)
    (h : ∀ F ∈ C.dtsets, t ∈ F.denote ↔ w ∈ F.denote) (n : Name) :
    (n, t) ∈ C.denote ↔ (n, w) ∈ C.denote := by
  induction C with
  | nameTest P => simp [Selector.denote]
  | ofType F => exact h F (by simp [Selector.dtsets])
  | not C ih => simp [Selector.denote, ih h]
  | and C D ihC ihD =>
    simp only [Selector.dtsets, List.mem_append] at h
    simp [Selector.denote, ihC fun F hF => h F (Or.inl hF), ihD fun F hF => h F (Or.inr hF)]
  | or C D ihC ihD =>
    simp only [Selector.dtsets, List.mem_append] at h
    simp [Selector.denote, ihC fun F hF => h F (Or.inl hF), ihD fun F hF => h F (Or.inr hF)]

/-- The single data types a list of data-type sets names. -/
def namedSingles (Fs : List DTSet) : List DType :=
  Fs.flatMap fun F => F.filterMap fun a =>
    match a with
    | .single t => some t
    | .kind _ => none

/-- A single data type in one of the sets is named. -/
theorem mem_namedSingles (Fs : List DTSet) (F : DTSet) (hF : F ∈ Fs) (u : DType)
    (hu : DTAtom.single u ∈ F) : u ∈ namedSingles Fs := by
  simp only [namedSingles, List.mem_flatMap, List.mem_filterMap]
  exact ⟨F, hF, _, hu, rfl⟩

/-- A data type of one kind that the sets do not name, when one exists. -/
noncomputable def kindRep (Fs : List DTSet) (k : Kind) : Option DType :=
  if h : ∃ t : DType, t.kind = k ∧ t ∉ namedSingles Fs then some (Classical.choose h) else none

/-- `T-9`. The witnesses: every named data type, and one unnamed data type of each kind. -/
noncomputable def witnesses (Fs : List DTSet) : List DType :=
  namedSingles Fs ++ Kind.all.filterMap (kindRep Fs)

/-- `T-9`. Every data type is treated like one of the witnesses by every set of the list. -/
theorem T9.witness_exists (Fs : List DTSet) (t : DType) :
    ∃ w ∈ witnesses Fs, ∀ F ∈ Fs, t ∈ F.denote ↔ w ∈ F.denote := by
  by_cases ht : t ∈ namedSingles Fs
  · exact ⟨t, List.mem_append_left _ ht, fun _ _ => Iff.rfl⟩
  · have hex : ∃ r : DType, r.kind = t.kind ∧ r ∉ namedSingles Fs := ⟨t, rfl, ht⟩
    obtain ⟨hrk, hrn⟩ := Classical.choose_spec hex
    refine ⟨Classical.choose hex, List.mem_append_right _ ?_, fun F hF => ?_⟩
    · refine List.mem_filterMap.mpr ⟨t.kind, Kind.mem_all _, ?_⟩
      simp [kindRep, hex]
    · simp only [DTSet.denote, Set.mem_ofPred_eq]
      refine exists_congr fun a => and_congr_right fun ha => ?_
      cases a with
      | kind k => simp [DTAtom.holds, hrk]
      | single u =>
        have hu := mem_namedSingles Fs F hF u ha
        simp only [DTAtom.holds]
        constructor
        · rintro rfl; exact absurd hu ht
        · rintro rfl; exact absurd hu hrn

/-- `T-9`, the reduction: containment between two selectors holds exactly when it holds at each
witness. -/
theorem T9.reduction (C D : Selector) :
    C.denote ⊆ D.denote ↔
      ∀ w ∈ witnesses (C.dtsets ++ D.dtsets), nameLang C w ⊆ nameLang D w := by
  constructor
  · intro h w _ n hn
    exact h hn
  · rintro h ⟨n, t⟩ hc
    obtain ⟨w, hw, hsame⟩ := T9.witness_exists (C.dtsets ++ D.dtsets) t
    have hC := (T9.same_types C t w (fun F hF => hsame F (List.mem_append_left _ hF)) n).1 hc
    have hD := h w hw hC
    exact (T9.same_types D t w (fun F hF => hsame F (List.mem_append_right _ hF)) n).2 hD

/-- `T-9`. At one data type, the names a selector picks form a regular language. -/
theorem T9.regular (C : Selector) (hC : C.MatchesRegular) (t : DType) :
    (toLang (nameLang C t)).IsRegular := by
  induction C with
  | nameTest P =>
    convert D11.regular P hC using 1
    ext n
    simp [nameLang, Selector.denote]
  | ofType F =>
    by_cases h : t ∈ F.denote
    · convert regular_univ using 1
      ext n
      simp [nameLang, Selector.denote, h]
    · convert regular_empty using 1
      ext n
      simp [nameLang, Selector.denote, h]
  | not C ih =>
    convert (ih hC).compl using 1
    ext n
    simp [nameLang, Selector.denote]
    rfl
  | and C D ihC ihD =>
    convert (ihC hC.1).inf (ihD hC.2) using 1
    ext n
    simp [nameLang, Selector.denote]
  | or C D ihC ihD =>
    convert (ihC hC.1).add (ihD hC.2) using 1
    ext n
    simp [nameLang, Selector.denote]
    rfl

/-- A finite automaton that accepts a word accepts one shorter than its number of states. -/
theorem short_word {σ : Type} [Fintype σ] (M : DFA Char σ) (x : Name) (hx : x ∈ M.accepts) :
    ∃ y ∈ M.accepts, y.length < Fintype.card σ := by
  have key : ∀ n, ∀ x ∈ M.accepts, x.length = n →
      ∃ y ∈ M.accepts, y.length < Fintype.card σ := by
    intro n
    induction n using Nat.strong_induction_on with
    | _ n ih =>
      intro x hx hlen
      by_cases hshort : x.length < Fintype.card σ
      · exact ⟨x, hx, hshort⟩
      · obtain ⟨a, b, c, rfl, _, hb, hpump⟩ := M.pumping_lemma hx (not_lt.mp hshort)
        have hmem : a ++ c ∈ ({a} * {b}∗ * {c} : Language Char) := by
          rw [Language.mem_mul]
          refine ⟨a ++ [], ?_, c, rfl, by simp⟩
          rw [Language.mem_mul]
          exact ⟨a, rfl, [], Language.nil_mem_kstar _, rfl⟩
        have hac : a ++ c ∈ M.accepts := hpump hmem
        have hlt : (a ++ c).length < n := by
          rw [← hlen]
          simp only [List.length_append]
          have := List.length_pos_of_ne_nil hb
          omega
        exact ih _ hlt (a ++ c) hac rfl
  exact key _ x hx rfl

/-- `T-9`, the finite check: containment between two regular languages holds exactly when it
holds for every word shorter than a bound. -/
theorem T9.finite_check (L₁ L₂ : Language Char) (h₁ : L₁.IsRegular) (h₂ : L₂.IsRegular) :
    ∃ k : ℕ, (L₁ ≤ L₂ ↔ ∀ x ∈ L₁, x.length < k → x ∈ L₂) := by
  obtain ⟨σ, _, M, hM⟩ := h₁.inf h₂.compl
  refine ⟨Fintype.card σ, fun h x hx _ => h hx, fun h x hx => ?_⟩
  by_contra hnot
  have hxM : x ∈ M.accepts := by rw [hM]; exact ⟨hx, hnot⟩
  obtain ⟨y, hy, hylen⟩ := short_word M x hxM
  rw [hM] at hy
  exact hy.2 (h y hy.1 hylen)

/-- A member of a list of natural numbers is at most its sum. -/
theorem le_sum_of_mem {l : List ℕ} {a : ℕ} (h : a ∈ l) : a ≤ l.sum := by
  induction l with
  | nil => simp at h
  | cons b l ih =>
    simp only [List.mem_cons] at h
    simp only [List.sum_cons]
    rcases h with rfl | h
    · omega
    · have := ih h
      omega

/-- `T-9`. Containment between two selectors is decided by a finite list of data types and a
bound on the length of names: it holds exactly when, for each listed data type, every name shorter
than the bound that the first selector picks, the second picks too. -/
theorem T9.decidable (C D : Selector) (hC : C.MatchesRegular) (hD : D.MatchesRegular) :
    ∃ (W : List DType) (k : ℕ), C.denote ⊆ D.denote ↔
      ∀ w ∈ W, ∀ x : Name, x.length < k → (x, w) ∈ C.denote → (x, w) ∈ D.denote := by
  let W := witnesses (C.dtsets ++ D.dtsets)
  have hk : ∀ w, ∃ k : ℕ, (toLang (nameLang C w) ≤ toLang (nameLang D w) ↔
      ∀ x ∈ toLang (nameLang C w), x.length < k → x ∈ toLang (nameLang D w)) := fun w =>
    T9.finite_check _ _ (T9.regular C hC w) (T9.regular D hD w)
  let bound := fun w => Classical.choose (hk w)
  let k := (W.map bound).sum
  have hbound : ∀ w ∈ W, bound w ≤ k := fun w hw =>
    le_sum_of_mem (List.mem_map.mpr ⟨w, hw, rfl⟩)
  refine ⟨W, k, ?_⟩
  rw [T9.reduction]
  constructor
  · intro h w hw x _ hx
    exact h w hw hx
  · intro h w hw
    have := (Classical.choose_spec (hk w)).2 fun x hx hlen =>
      h w hw x (lt_of_lt_of_le hlen (hbound w hw)) hx
    exact this

end Spec
