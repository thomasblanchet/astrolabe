import Mathlib.Data.List.Basic

/-!
# What several data types share

An answer at an open column is the data type that the answers at each data type of the sample
share. A data type is a kind with its parameters and its inner data types, as `List(Int8)` or
`Datetime("ms", None)`. A shared type can leave the parameters of a kind open, as `Datetime`, or
be `Unchecked`.

The theorem of this file (`T-26`) says that each answer fits what it shares with another, so the
join of the answers is an upper bound of each of them.
-/

namespace Spec.Generalize

/-- A data type with holes over the kinds `K` and the parameters `P`. -/
inductive Pat (K P : Type) where
  /-- Every data type. -/
  | unchecked : Pat K P
  /-- Every data type of one kind, whatever its parameters and its inner data types. -/
  | kind : K → Pat K P
  /-- A kind with its parameters and its inner data types, as `List(Int8)`. -/
  | full : K → P → List (Pat K P) → Pat K P

variable {K P : Type} [DecidableEq K] [DecidableEq P]

mutual
/-- A data type fits a data type with holes. -/
def Fits : Pat K P → Pat K P → Prop
  | _, .unchecked => True
  | .kind k, .kind k' => k = k'
  | .full k _ _, .kind k' => k = k'
  | .full k p inner, .full k' p' inner' => k = k' ∧ p = p' ∧ FitsAll inner inner'
  | _, _ => False

/-- Each data type of a list fits the data type at its place in another list of the same length. -/
def FitsAll : List (Pat K P) → List (Pat K P) → Prop
  | [], [] => True
  | a :: rest, b :: rest' => Fits a b ∧ FitsAll rest rest'
  | _, _ => False
end

mutual
/-- What two data types share: their parts where they agree, the kind alone where a parameter or
the number of inner data types differs, and `Unchecked` where the kinds differ. -/
def join : Pat K P → Pat K P → Pat K P
  | .full k p inner, .full k' p' inner' =>
      if k = k' then
        if p = p' ∧ inner.length = inner'.length then .full k p (joinAll inner inner')
        else .kind k
      else .unchecked
  | .full k _ _, .kind k' => if k = k' then .kind k else .unchecked
  | .kind k, .full k' _ _ => if k = k' then .kind k else .unchecked
  | .kind k, .kind k' => if k = k' then .kind k else .unchecked
  | _, _ => .unchecked

/-- What two lists of data types share, place by place. -/
def joinAll : List (Pat K P) → List (Pat K P) → List (Pat K P)
  | a :: rest, b :: rest' => join a b :: joinAll rest rest'
  | _, _ => []
end

/-- `T-26`. A data type fits what it shares with another. -/
theorem fits_join_left (a : Pat K P) : ∀ b, Fits a (join a b) := by
  induction a using Pat.rec (motive_2 := fun (as : List (Pat K P)) =>
    ∀ bs : List (Pat K P), as.length = bs.length → FitsAll as (joinAll as bs)) with
  | unchecked => intro b; cases b <;> simp [join, Fits]
  | kind k =>
      intro b
      cases b with
      | unchecked => simp [join, Fits]
      | kind k' => rw [join]; split <;> simp_all [Fits]
      | full k' _ _ => rw [join]; split <;> simp_all [Fits]
  | full k p inner ih =>
      intro b
      cases b with
      | unchecked => simp [join, Fits]
      | kind k' => rw [join]; split <;> simp_all [Fits]
      | full k' p' inner' =>
          rw [join]
          split
          · split
            · rename_i hp
              rw [Fits]
              exact ⟨rfl, rfl, ih inner' hp.2⟩
            · simp_all [Fits]
          · simp [Fits]
  | nil =>
      rename_i bs h
      cases bs with
      | nil => simp [joinAll, FitsAll]
      | cons _ _ => simp at h
  | cons a rest iha ihrest =>
      rename_i bs h
      cases bs with
      | nil => simp at h
      | cons b rest' =>
          rw [joinAll, FitsAll]
          exact ⟨iha b, ihrest rest' (by simpa using h)⟩

/-- `T-26`, the other side. A data type fits what another shares with it. -/
theorem fits_join_right (b : Pat K P) : ∀ a, Fits b (join a b) := by
  induction b using Pat.rec (motive_2 := fun (bs : List (Pat K P)) =>
    ∀ as : List (Pat K P), as.length = bs.length → FitsAll bs (joinAll as bs)) with
  | unchecked => intro a; cases a <;> simp [join, Fits]
  | kind k =>
      intro a
      cases a with
      | unchecked => simp [join, Fits]
      | kind k' => rw [join]; split <;> simp_all [Fits]
      | full k' _ _ => rw [join]; split <;> simp_all [Fits]
  | full k p inner ih =>
      intro a
      cases a with
      | unchecked => simp [join, Fits]
      | kind k' => rw [join]; split <;> simp_all [Fits]
      | full k' p' inner' =>
          rw [join]
          split
          · split
            · rename_i hk hp
              rw [Fits]
              exact ⟨hk.symm, hp.1.symm, ih inner' hp.2⟩
            · rename_i hk _
              rw [Fits]
              exact hk.symm
          · simp [Fits]
  | nil =>
      rename_i as h
      cases as with
      | nil => simp [joinAll, FitsAll]
      | cons _ _ => simp at h
  | cons b rest ihb ihrest =>
      rename_i as h
      cases as with
      | nil => simp at h
      | cons a rest' =>
          rw [joinAll, FitsAll]
          exact ⟨ihb a, ihrest rest' (by simpa using h)⟩

/-- What a data type fits, it fits after a join with another. -/
theorem fits_join_of_fits (d : Pat K P) : ∀ x c, Fits d x → Fits d (join x c) := by
  induction d using Pat.rec (motive_2 := fun (ds : List (Pat K P)) =>
    ∀ xs cs : List (Pat K P), FitsAll ds xs → xs.length = cs.length →
      FitsAll ds (joinAll xs cs)) with
  | unchecked =>
      intro x c h
      cases x with
      | unchecked => cases c <;> simp [join, Fits]
      | kind _ => simp [Fits] at h
      | full _ _ _ => simp [Fits] at h
  | kind k =>
      intro x c h
      cases x with
      | unchecked => cases c <;> simp [join, Fits]
      | kind k' =>
          rw [Fits] at h
          subst h
          cases c with
          | unchecked => simp [join, Fits]
          | kind k'' => rw [join]; split <;> simp_all [Fits]
          | full k'' _ _ => rw [join]; split <;> simp_all [Fits]
      | full _ _ _ => simp [Fits] at h
  | full k p inner ih =>
      intro x c h
      cases x with
      | unchecked => cases c <;> simp [join, Fits]
      | kind k' =>
          rw [Fits] at h
          subst h
          cases c with
          | unchecked => simp [join, Fits]
          | kind k'' => rw [join]; split <;> simp_all [Fits]
          | full k'' _ _ => rw [join]; split <;> simp_all [Fits]
      | full k' p' inner' =>
          rw [Fits] at h
          obtain ⟨rfl, rfl, hinner⟩ := h
          cases c with
          | unchecked => simp [join, Fits]
          | kind k'' => rw [join]; split <;> simp_all [Fits]
          | full k'' p'' inner'' =>
              rw [join]
              split
              · split
                · rename_i hp
                  rw [Fits]
                  exact ⟨rfl, rfl, ih inner' inner'' hinner hp.2⟩
                · rw [Fits]
              · simp [Fits]
  | nil =>
      rename_i xs cs h hlen
      cases xs with
      | nil => cases cs <;> simp_all [joinAll, FitsAll]
      | cons _ _ => simp [FitsAll] at h
  | cons d rest ihd ihrest =>
      rename_i xs cs h hlen
      cases xs with
      | nil => simp [FitsAll] at h
      | cons x xrest =>
          cases cs with
          | nil => simp at hlen
          | cons c crest =>
              rw [FitsAll] at h
              rw [joinAll, FitsAll]
              exact ⟨ihd x c h.1, ihrest xrest crest h.2 (by simpa using hlen)⟩

omit [DecidableEq K] [DecidableEq P] in
/-- A data type fits itself. -/
theorem fits_refl (a : Pat K P) : Fits a a := by
  induction a using Pat.rec (motive_2 := fun (as : List (Pat K P)) => FitsAll as as) with
  | unchecked => simp [Fits]
  | kind k => simp [Fits]
  | full k p inner ih => rw [Fits]; exact ⟨rfl, rfl, ih⟩
  | nil => simp [FitsAll]
  | cons a rest iha ihrest => rw [FitsAll]; exact ⟨iha, ihrest⟩

/-- What a data type fits, it fits after the joins with each data type of a list. -/
theorem fits_foldl (d : Pat K P) :
    ∀ (rest : List (Pat K P)) (x : Pat K P), Fits d x → Fits d (rest.foldl join x)
  | [], _, h => h
  | c :: rest, x, h => by
      rw [List.foldl_cons]
      exact fits_foldl d rest (join x c) (fits_join_of_fits d x c h)

/-- What the answers at the data types of a sample share: the join of all of them. -/
def generalize (first : Pat K P) (rest : List (Pat K P)) : Pat K P :=
  rest.foldl join first

/-- `T-26`. Each answer fits what all the answers share. -/
theorem fits_generalize (first : Pat K P) (rest : List (Pat K P)) :
    ∀ d ∈ first :: rest, Fits d (generalize first rest) := by
  induction rest generalizing first with
  | nil =>
      intro d hd
      simp only [List.mem_singleton] at hd
      subst hd
      exact fits_refl d
  | cons b rest ih =>
      intro d hd
      unfold generalize at ih ⊢
      rw [List.foldl_cons]
      rcases List.mem_cons.mp hd with h | h
      · subst h
        exact fits_foldl d rest (join d b) (fits_join_left d b)
      · rcases List.mem_cons.mp h with h | h
        · subst h
          exact fits_foldl d rest (join first d) (fits_join_right d first)
        · exact ih (join first b) d (List.mem_cons_of_mem _ h)

end Spec.Generalize
