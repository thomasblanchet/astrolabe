import Mathlib.Data.Set.Basic
import Mathlib.Data.Fintype.Basic

/-!
# Layer 0: the objects (`D-1` to `D-5`, `D-35`)

Names, data types, kinds, column sets and data-type sets. Every later layer builds on these.
-/

namespace Spec

/-- `D-1`. A name is a finite string. A list of characters gives the append lemmas directly. -/
abbrev Name := List Char

/-- The time unit of `Datetime` and `Duration`. -/
inductive TimeUnit
  | ns
  | us
  | ms
  deriving DecidableEq, Repr

/-- The polars data types that carry no parameter. -/
inductive Scalar
  | boolean
  | int8
  | int16
  | int32
  | int64
  | int128
  | uint8
  | uint16
  | uint32
  | uint64
  | uint128
  | float16
  | float32
  | float64
  | string
  | binary
  | date
  | time
  | null
  | object
  | categorical
  deriving DecidableEq, Repr

/-- Every parameterless data type, listed once. -/
def Scalar.all : List Scalar :=
  [.boolean, .int8, .int16, .int32, .int64, .int128, .uint8, .uint16, .uint32, .uint64, .uint128,
    .float16, .float32, .float64, .string, .binary, .date, .time, .null, .object, .categorical]

/-- The list of parameterless data types is complete. -/
theorem Scalar.mem_all (s : Scalar) : s ∈ Scalar.all := by
  cases s <;> simp [Scalar.all]

/-- `D-35`. A kind is a data type with its parameters erased. There are finitely many. -/
inductive Kind
  | scalar (s : Scalar)
  | decimal
  | datetime
  | duration
  | enum
  | list
  | array
  | struct
  deriving DecidableEq, Repr

/-- Every kind, listed once. -/
def Kind.all : List Kind :=
  Scalar.all.map .scalar ++ [.decimal, .datetime, .duration, .enum, .list, .array, .struct]

/-- The list of kinds is complete. -/
theorem Kind.mem_all (k : Kind) : k ∈ Kind.all := by
  cases k <;> simp [Kind.all, Scalar.mem_all]

/-- `D-2`. The kinds are finite. -/
instance : Fintype Kind :=
  Fintype.ofList Kind.all Kind.mem_all

/-- `D-2`. A data type. Data types nest without bound, so the type is infinite. Struct fields nest
through a list, so the inductive stays free of quotients. -/
inductive DType
  | scalar (s : Scalar)
  | decimal (precision scale : ℕ)
  | datetime (unit : TimeUnit) (zone : Option String)
  | duration (unit : TimeUnit)
  | enum (categories : List String)
  | list (inner : DType)
  | array (inner : DType) (width : ℕ)
  | struct (fields : List (Name × DType))

/-- `D-35`. The kind of a data type: the data type with its parameters erased. -/
def DType.kind : DType → Kind
  | .scalar s => .scalar s
  | .decimal _ _ => .decimal
  | .datetime _ _ => .datetime
  | .duration _ => .duration
  | .enum _ => .enum
  | .list _ => .list
  | .array _ _ => .array
  | .struct _ => .struct

/-- One data type of each kind, so every kind holds a data type. -/
def Kind.witness : Kind → DType
  | .scalar s => .scalar s
  | .decimal => .decimal 38 0
  | .datetime => .datetime .us none
  | .duration => .duration .us
  | .enum => .enum []
  | .list => .list (.scalar .null)
  | .array => .array (.scalar .null) 1
  | .struct => .struct []

/-- The witness of a kind is of that kind. -/
theorem Kind.witness_kind (k : Kind) : k.witness.kind = k := by
  cases k <;> rfl

/-- `D-4` and `D-5`. A column set is a set of columns `n : t`. It is a column set in the sense of
`D-5` when it is functional: no name carries two data types. -/
abbrev ColumnSet := Set (Name × DType)

/-- A column set is functional when every name carries at most one data type. -/
def Functional (X : ColumnSet) : Prop :=
  ∀ n t u, (n, t) ∈ X → (n, u) ∈ X → t = u

/-- One member of a data-type set: a whole kind, or one data type. -/
inductive DTAtom
  | kind (k : Kind)
  | single (t : DType)

/-- A data type is in the set one atom denotes. -/
def DTAtom.holds : DTAtom → DType → Prop
  | .kind k, t => t.kind = k
  | .single u, t => t = u

/-- `D-3`. A data-type set, written as a finite union of kinds and single data types. -/
abbrev DTSet := List DTAtom

/-- `D-3`. The data types a data-type set denotes. -/
def DTSet.denote (F : DTSet) : Set DType :=
  {t | ∃ a ∈ F, a.holds t}

/-- A subset of a functional column set is functional. -/
theorem Functional.mono {X Y : ColumnSet} (hY : Functional Y) (hXY : X ⊆ Y) : Functional X :=
  fun n t u ht hu => hY n t u (hXY ht) (hXY hu)

end Spec
