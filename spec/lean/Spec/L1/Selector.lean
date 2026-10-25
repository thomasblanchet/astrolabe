import Spec.L1.Absence

/-!
# Layer 1: selectors and type sets (`D-12`, `D-13`, `D-15`)

A selector decides a whole column. A type set records the data types a remainder can hold.
-/

namespace Spec

/-- `D-12`. The selector language. -/
inductive Selector
  | nameTest (P : Absence)
  | ofType (F : DTSet)
  | not (C : Selector)
  | and (C D : Selector)
  | or (C D : Selector)

/-- `D-13`. The columns a selector picks. -/
def Selector.denote : Selector → Set (Name × DType)
  | .nameTest P => {c | c.1 ∈ P.denote}
  | .ofType F => {c | c.2 ∈ F.denote}
  | .not C => (C.denote)ᶜ
  | .and C D => C.denote ∩ D.denote
  | .or C D => C.denote ∪ D.denote

/-- A selector reads no name when it holds no name test. The check is syntactic: a name test
that holds of every name or of none still reads a name. -/
def Selector.readsNoName : Selector → Bool
  | .nameTest _ => false
  | .ofType _ => true
  | .not C => C.readsNoName
  | .and C D => C.readsNoName && D.readsNoName
  | .or C D => C.readsNoName && D.readsNoName

/-- `D-15`. A type set, in the forms the rules of section 7 produce. -/
inductive TypeSet
  | univ
  | empty
  | ofSet (F : DTSet)
  | single (U : DType)
  | union (A B : TypeSet)
  | inter (A B : TypeSet)
  | diff (A B : TypeSet)

/-- `D-15`. The data types a type set denotes. -/
def TypeSet.denote : TypeSet → Set DType
  | .univ => Set.univ
  | .empty => ∅
  | .ofSet F => F.denote
  | .single U => {U}
  | .union A B => A.denote ∪ B.denote
  | .inter A B => A.denote ∩ B.denote
  | .diff A B => A.denote \ B.denote

end Spec
