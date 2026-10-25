import Spec.L1.Absence
import Mathlib.Computability.MyhillNerode
import Mathlib.Computability.NFA
import Mathlib.Data.List.Infix

/-!
# Layer 4: every absence rule denotes a regular set (`D-11`)

A language is regular when a finite automaton decides it. Mathlib states this for `Language Char`
and proves it closed under union, intersection, complement and reversal. The Myhill–Nerode theorem
turns every other closure into a count of left quotients.
-/

namespace Spec

open Language

/-- A set of names, read as a language over characters. -/
def toLang (S : Set Name) : Language Char := S

@[simp] theorem mem_toLang (S : Set Name) (w : Name) : w ∈ toLang S ↔ w ∈ S := Iff.rfl

/-- A language all of whose left quotients lie in a finite set is regular. -/
theorem regular_of_quotients_subset (L : Language Char) (Q : Set (Language Char))
    (hQ : Q.Finite) (h : ∀ x, L.leftQuotient x ∈ Q) : L.IsRegular :=
  IsRegular.of_finite_range_leftQuotient (hQ.subset (by rintro _ ⟨x, rfl⟩; exact h x))

/-- The empty language is regular. -/
theorem regular_empty : (∅ : Set Name) |> toLang |>.IsRegular :=
  regular_of_quotients_subset _ {toLang ∅} (Set.finite_singleton _) fun x => by
    simp only [Set.mem_singleton_iff]
    ext y
    simp [mem_leftQuotient]

/-- The language of every name is regular. -/
theorem regular_univ : (Set.univ : Set Name) |> toLang |>.IsRegular :=
  regular_of_quotients_subset _ {toLang Set.univ} (Set.finite_singleton _) fun x => by
    simp only [Set.mem_singleton_iff]
    ext y
    simp [mem_leftQuotient]

/-- The names made of a fixed word followed by a name of `Q`. -/
def prefixWord (s : Name) (Q : Set Name) : Set Name :=
  {w | ∃ m ∈ Q, w = s ++ m}

/-- The left quotient of `prefixWord s Q` by a word that `s` starts with. -/
theorem prefixWord_quotient_long (s y : Name) (Q : Set Name) :
    (toLang (prefixWord s Q)).leftQuotient (s ++ y) = (toLang Q).leftQuotient y := by
  ext v
  simp only [mem_leftQuotient, mem_toLang, prefixWord, Set.mem_ofPred_eq, List.append_assoc]
  constructor
  · rintro ⟨m, hm, h⟩
    rw [List.append_cancel_left h]
    exact hm
  · intro h
    exact ⟨_, h, rfl⟩

/-- The left quotient of `prefixWord s Q` by a word that is a prefix of `s`. -/
theorem prefixWord_quotient_short (x r : Name) (Q : Set Name) :
    (toLang (prefixWord (x ++ r) Q)).leftQuotient x = toLang (prefixWord r Q) := by
  ext v
  simp only [mem_leftQuotient, mem_toLang, prefixWord, Set.mem_ofPred_eq, List.append_assoc]
  constructor
  · rintro ⟨m, hm, h⟩
    exact ⟨m, hm, List.append_cancel_left h⟩
  · rintro ⟨m, hm, rfl⟩
    exact ⟨m, hm, rfl⟩

/-- The left quotient of `prefixWord s Q` by a word that neither starts `s` nor is a prefix of it
is empty. -/
theorem prefixWord_quotient_other (s x : Name) (Q : Set Name) (h₁ : ¬ s <+: x) (h₂ : ¬ x <+: s) :
    (toLang (prefixWord s Q)).leftQuotient x = toLang ∅ := by
  ext v
  simp only [mem_leftQuotient, mem_toLang, prefixWord, Set.mem_ofPred_eq, Set.mem_empty_iff_false,
    iff_false, not_exists, not_and]
  intro m _ h
  rcases List.append_eq_append_iff.mp h with ⟨a, rfl, _⟩ | ⟨c, rfl, _⟩
  · exact h₂ ⟨a, rfl⟩
  · exact h₁ ⟨c, rfl⟩

/-- Prefixing a fixed word keeps a language regular. -/
theorem regular_prefixWord (s : Name) (Q : Set Name) (hQ : (toLang Q).IsRegular) :
    (toLang (prefixWord s Q)).IsRegular := by
  let quotients : Set (Language Char) :=
    Set.range (toLang Q).leftQuotient ∪
      ((fun r => toLang (prefixWord r Q)) '' {r | r ∈ s.tails}) ∪ {toLang ∅}
  refine regular_of_quotients_subset _ quotients ?_ fun x => ?_
  · refine ((hQ.finite_range_leftQuotient).union ?_).union (Set.finite_singleton _)
    exact (List.finite_toSet s.tails).image _
  · by_cases h₁ : s <+: x
    · obtain ⟨y, rfl⟩ := h₁
      rw [prefixWord_quotient_long]
      exact Or.inl (Or.inl ⟨y, rfl⟩)
    · by_cases h₂ : x <+: s
      · obtain ⟨r, rfl⟩ := h₂
        rw [prefixWord_quotient_short]
        exact Or.inl (Or.inr ⟨r, (List.mem_tails r (x ++ r)).mpr ⟨x, rfl⟩, rfl⟩)
      · rw [prefixWord_quotient_other s x Q h₁ h₂]
        exact Or.inr rfl

/-- The names made of a name of `L` followed by anything. -/
def appendAny (L : Set Name) : Set Name :=
  {w | ∃ x ∈ L, ∃ y, w = x ++ y}

open Classical in
/-- The left quotient of `appendAny L` by a word: every name, when a prefix of the word is in `L`;
otherwise `appendAny` of `L`'s own left quotient. -/
theorem appendAny_quotient (L : Set Name) (w : Name) :
    (toLang (appendAny L)).leftQuotient w =
      if ∃ x ∈ L, x <+: w then toLang Set.univ
      else toLang (appendAny ((toLang L).leftQuotient w)) := by
  split_ifs with h
  · obtain ⟨x, hx, c, rfl⟩ := h
    ext v
    simp only [mem_leftQuotient, mem_toLang, appendAny, Set.mem_ofPred_eq, Set.mem_univ, iff_true]
    exact ⟨x, hx, c ++ v, by simp⟩
  · ext v
    simp only [mem_leftQuotient, mem_toLang, appendAny, Set.mem_ofPred_eq]
    constructor
    · rintro ⟨x, hx, y, hxy⟩
      rcases List.append_eq_append_iff.mp hxy with ⟨a, rfl, rfl⟩ | ⟨c, rfl, rfl⟩
      · exact ⟨a, hx, y, rfl⟩
      · exact absurd ⟨x, hx, c, rfl⟩ h
    · rintro ⟨x', hx', y, rfl⟩
      exact ⟨w ++ x', hx', y, by simp⟩

open Classical in
/-- Appending anything keeps a language regular. -/
theorem regular_appendAny (L : Set Name) (hL : (toLang L).IsRegular) :
    (toLang (appendAny L)).IsRegular := by
  let quotients : Set (Language Char) :=
    {toLang Set.univ} ∪ (fun M => toLang (appendAny M)) '' Set.range (toLang L).leftQuotient
  refine regular_of_quotients_subset _ quotients
    ((Set.finite_singleton _).union (hL.finite_range_leftQuotient.image _)) fun w => ?_
  rw [appendAny_quotient]
  split_ifs
  · exact Or.inl rfl
  · exact Or.inr ⟨_, ⟨w, rfl⟩, rfl⟩

/-- The language holding only the empty name is regular. -/
theorem regular_nil : (toLang {[]}).IsRegular :=
  regular_of_quotients_subset _ {toLang {[]}, toLang ∅} (Set.toFinite _) fun w => by
    cases w with
    | nil => exact Or.inl rfl
    | cons c w =>
      refine Or.inr ?_
      ext v
      simp [mem_leftQuotient]

/-- A finite list of names denotes a regular language. -/
theorem regular_list (S : List Name) : (toLang {n | n ∈ S}).IsRegular := by
  induction S with
  | nil => simpa using regular_empty
  | cons s S ih =>
    have h := (regular_prefixWord s {[]} regular_nil).add ih
    convert h using 1
    ext w
    simp [prefixWord, Language.mem_add]

/-- The names that start with `s` form a regular language. -/
theorem regular_startsWith (s : Name) : (toLang {n | s <+: n}).IsRegular := by
  convert regular_prefixWord s Set.univ regular_univ using 1
  ext w
  simp [prefixWord, List.IsPrefix, eq_comm]

/-- The names that end with `s` form a regular language. -/
theorem regular_endsWith (s : Name) : (toLang {n | s <:+ n}).IsRegular := by
  convert (regular_startsWith s.reverse).reverse using 1
  ext w
  simp only [mem_toLang, Language.reverse]
  show s <:+ w ↔ s.reverse <+: w.reverse
  exact List.reverse_prefix.symm

/-- The names that contain `s` form a regular language. -/
theorem regular_contains (s : Name) : (toLang {n | s <:+: n}).IsRegular := by
  convert regular_appendAny _ (regular_endsWith s) using 1
  ext w
  simp only [mem_toLang, Set.mem_ofPred_eq, appendAny]
  constructor
  · rintro ⟨a, b, rfl⟩
    exact ⟨a ++ s, ⟨a, rfl⟩, b, rfl⟩
  · rintro ⟨x, ⟨a, rfl⟩, b, rfl⟩
    exact ⟨a, b, rfl⟩

/-- A name of `P` followed by a fixed suffix forms a regular language when `P` does. -/
theorem regular_afterStripping (s : Name) (P : Set Name) (hP : (toLang P).IsRegular) :
    (toLang {w | ∃ n ∈ P, w = n ++ s}).IsRegular := by
  convert (regular_prefixWord s.reverse (toLang P).reverse hP.reverse).reverse using 1
  ext w
  simp only [mem_toLang, Set.mem_ofPred_eq, Language.reverse, prefixWord]
  constructor
  · rintro ⟨n, hn, rfl⟩
    exact ⟨n.reverse, by simpa using hn, by simp⟩
  · rintro ⟨m, hm, h⟩
    refine ⟨m.reverse, hm, ?_⟩
    rw [← List.reverse_reverse w, h]
    simp

/-- `D-11`. Every `matching` in a rule decides a regular set. -/
def Absence.MatchesRegular : Absence → Prop
  | .matching m => (toLang m.denote).IsRegular
  | .not P => P.MatchesRegular
  | .afterStripping _ P => P.MatchesRegular
  | .beforeStripping _ P => P.MatchesRegular
  | .and P Q => P.MatchesRegular ∧ Q.MatchesRegular
  | .or P Q => P.MatchesRegular ∧ Q.MatchesRegular
  | _ => True

/-- `D-11`. Every absence rule denotes a regular set, when each `matching` in it does. -/
theorem D11.regular (P : Absence) (h : P.MatchesRegular) : (toLang P.denote).IsRegular := by
  induction P with
  | never => exact regular_empty
  | always => exact regular_univ
  | oneOf S => exact regular_list S
  | startsWith s => exact regular_startsWith s
  | endsWith s => exact regular_endsWith s
  | contains s => exact regular_contains s
  | matching m => exact h
  | not P ih => exact (ih h).compl
  | afterStripping s P ih => exact regular_afterStripping s _ (ih h)
  | beforeStripping s P ih =>
    have := regular_prefixWord s _ (ih h)
    convert this using 1
    ext w
    simp [prefixWord, Absence.denote]
  | and P Q ihP ihQ => exact (ihP h.1).inf (ihQ h.2)
  | or P Q ihP ihQ => exact (ihP h.1).add (ihQ h.2)

/-- A left quotient of a regular language is regular. -/
theorem regular_leftQuotient (L : Language Char) (hL : L.IsRegular) (x : Name) :
    (L.leftQuotient x).IsRegular := by
  refine IsRegular.of_finite_range_leftQuotient (hL.finite_range_leftQuotient.subset ?_)
  rintro _ ⟨y, rfl⟩
  exact ⟨x ++ y, leftQuotient_append L x y⟩

/-- `D-11`. The translation of `D-23` keeps a rule regular: the names `n` with
`pre ++ n ++ suffix` in a regular language form a regular language. -/
theorem D11.preimage_regular (L : Set Name) (hL : (toLang L).IsRegular) (pre suffix : Name) :
    (toLang {n | pre ++ (n ++ suffix) ∈ L}).IsRegular := by
  have hq := regular_leftQuotient _ hL pre
  have hr := (regular_leftQuotient _ hq.reverse suffix.reverse).reverse
  convert hr using 1
  ext n
  show pre ++ (n ++ suffix) ∈ L ↔ pre ++ (suffix.reverse ++ n.reverse).reverse ∈ L
  simp

end Spec
