import Mathlib.Data.Finset.Basic
import Mathlib.Data.Set.Basic
import Mathlib.Logic.Function.Basic

/-!
# The evaluator of trees

An expression is a tree of nodes. A leaf reads a column of the frame or holds a literal, and a node
applies one operation of polars to the answers of its inputs. Polars computes a state for each node
from the states of its inputs: the types of its three passes, its row form, and the facts of its data
that later nodes read. The evaluator of the checker does the same with the rules of each node.

The theorem of this file (`T-25`) says that the evaluator gives the answer of polars on the whole
tree, where each rule gives the answer of polars on the states of its inputs. It holds under one
hypothesis: polars computes the state of a node only from the states of its inputs (`locality`).
The verification of random trees measures that hypothesis.

The second half of the file lifts the evaluator to columns whose data type is a set (`D-41`), and
proves that the lifting is sound, monotone, and blind to the columns that a tree does not read
(`T-21`, `T-22`).
-/

namespace Spec.Node

/-- A tree of nodes over column names `N`, literal values `V` and operations `Op`. -/
inductive Tree (N V Op : Type) where
  /-- A read of the column of one name. -/
  | column : N → Tree N V Op
  /-- A literal of one value. -/
  | literal : V → Tree N V Op
  /-- An operation over its inputs, in order. -/
  | node : Op → List (Tree N V Op) → Tree N V Op

variable {N V Op T S : Type}

/-- What polars does, and the hypothesis that it works node by node.

`answer cols t` is the state that polars gives the tree `t` over a frame whose columns have the data
types `cols`, or `none` where polars refuses. `column`, `literal` and `node` give the state of a leaf
and of one node from the states of its inputs. -/
structure Polars (N V Op T S : Type) where
  /-- The state of the whole tree, as polars computes it. -/
  answer : (N → T) → Tree N V Op → Option S
  /-- The state of a column of a data type. -/
  column : T → Option S
  /-- The state of a literal. -/
  literal : V → Option S
  /-- The state of one node, from the states of its inputs. -/
  node : Op → List S → Option S
  /-- `M-20`. Polars types a node only from the states of its inputs. Measured by the verification
  of random trees, `spec/tools/rules/tree_check.py`, and by
  `test_the_evaluation_of_a_tree_gives_the_answers_of_polars` in
  `spec/tests/test_rules_tree_evaluation.py`. -/
  locality : ∀ cols t, answer cols t =
    match t with
    | .column n => column (cols n)
    | .literal v => literal v
    | .node op inputs => (inputs.mapM (answer cols)).bind (node op)

/-- The rules of the checker: the state of a leaf, and of one node from the states of its inputs. -/
structure Rules (V Op T S : Type) where
  /-- The state of a column of a data type. -/
  column : T → Option S
  /-- The state of a literal. -/
  literal : V → Option S
  /-- The state of one node, from the states of its inputs. -/
  node : Op → List S → Option S

mutual
/-- The evaluator: the state of each node from the states of its inputs, from the leaves up. -/
def evaluate (rules : Rules V Op T S) (cols : N → T) : Tree N V Op → Option S
  | .column n => rules.column (cols n)
  | .literal v => rules.literal v
  | .node op inputs => (evaluateAll rules cols inputs).bind (rules.node op)

/-- The states of a list of inputs, in order, or `none` where one of them has none. -/
def evaluateAll (rules : Rules V Op T S) (cols : N → T) : List (Tree N V Op) → Option (List S)
  | [] => some []
  | t :: rest => (evaluate rules cols t).bind fun s => (evaluateAll rules cols rest).map (s :: ·)
end

/-- The rules are correct where each gives the state of polars on every state of its inputs. The
tests of `spec/tests/test_definitions_*.py` and the replay of the node census measure each rule. -/
structure Correct (polars : Polars N V Op T S) (rules : Rules V Op T S) : Prop where
  column : ∀ ty, rules.column ty = polars.column ty
  literal : ∀ v, rules.literal v = polars.literal v
  node : ∀ op states, rules.node op states = polars.node op states

/-- `T-25`. With correct rules, the evaluator gives the state of polars on the whole tree. -/
theorem T25.evaluate_eq_answer (polars : Polars N V Op T S) (rules : Rules V Op T S)
    (correct : Correct polars rules) (cols : N → T) :
    ∀ t, evaluate rules cols t = polars.answer cols t := by
  intro t
  induction t using Tree.rec (motive_2 := fun inputs =>
    evaluateAll rules cols inputs = inputs.mapM (polars.answer cols)) with
  | column n => rw [polars.locality]; simp [evaluate, correct.column]
  | literal v => rw [polars.locality]; simp [evaluate, correct.literal]
  | node op inputs ih =>
      rw [polars.locality]
      simp only [evaluate, ih]
      cases inputs.mapM (polars.answer cols) <;> simp [correct.node]
  | nil => rfl
  | cons t rest iht ihrest =>
      simp only [evaluateAll, iht, ihrest, List.mapM_cons]
      cases polars.answer cols t <;> cases rest.mapM (polars.answer cols) <;> rfl

section Lifted

variable [DecidableEq N]

mutual
/-- The names of the columns that a tree reads. -/
def reads : Tree N V Op → Finset N
  | .column n => {n}
  | .literal _ => ∅
  | .node _ inputs => readsAll inputs

/-- The names of the columns that a list of trees reads. -/
def readsAll : List (Tree N V Op) → Finset N
  | [] => ∅
  | t :: rest => reads t ∪ readsAll rest
end

/-- The evaluator reads a column only where the tree reads its name. -/
theorem evaluate_congr (rules : Rules V Op T S) (cols cols' : N → T) :
    ∀ t : Tree N V Op, (∀ n ∈ reads t, cols n = cols' n) →
      evaluate rules cols t = evaluate rules cols' t := by
  intro t
  induction t using Tree.rec (motive_2 := fun inputs =>
    (∀ n ∈ readsAll inputs, cols n = cols' n) →
      evaluateAll rules cols inputs = evaluateAll rules cols' inputs) with
  | column n =>
      intro h
      simp only [evaluate]
      rw [h n (by simp [reads])]
  | literal v => intro _; rfl
  | node op inputs ih =>
      intro h
      simp only [evaluate]
      rw [ih (by simpa [reads] using h)]
  | nil => rfl
  | cons t rest iht ihrest =>
      rename_i h
      simp only [evaluateAll]
      rw [iht (fun n hn => h n (by simp [readsAll, hn])),
        ihrest (fun n hn => h n (by simp [readsAll, hn]))]

omit [DecidableEq N] in
/-- `D-41`. The states that a tree gives where each column holds a data type of a set: the state at
each choice of one member of each set. An open column, whose data type has holes, is such a set, and
`Unchecked` is the set of every data type. -/
def lifted (rules : Rules V Op T S) (X : N → Set T) (t : Tree N V Op) : Set S :=
  {s | ∃ cols : N → T, (∀ n, cols n ∈ X n) ∧ evaluate rules cols t = some s}

omit [DecidableEq N] in
/-- `D-41`. The lifted states hold the state at every choice of the members. -/
theorem D41.lift_sound (rules : Rules V Op T S) (X : N → Set T) (cols : N → T)
    (h : ∀ n, cols n ∈ X n) (t : Tree N V Op) (s : S) (he : evaluate rules cols t = some s) :
    s ∈ lifted rules X t :=
  ⟨cols, h, he⟩

omit [DecidableEq N] in
/-- `T-22`. Widening the data types of the columns widens the states. -/
theorem T22.monotone (rules : Rules V Op T S) (X Y : N → Set T) (h : ∀ n, X n ⊆ Y n)
    (t : Tree N V Op) : lifted rules X t ⊆ lifted rules Y t := by
  rintro s ⟨cols, hcols, he⟩
  exact ⟨cols, fun n => h n (hcols n), he⟩

/-- `T-22`. A tree with a state at some choice keeps one after a column becomes `Unchecked`. -/
theorem T22.no_new_error (rules : Rules V Op T S) (X : N → Set T) (n : N) (t : Tree N V Op)
    (h : (lifted rules X t).Nonempty) :
    (lifted rules (Function.update X n Set.univ) t).Nonempty := by
  apply h.mono
  apply T22.monotone
  intro m
  by_cases hm : m = n
  · subst hm
    simp
  · simp [Function.update_of_ne hm]

/-- `T-21`. Replacing the data types of a column that the tree does not read changes no state. -/
theorem T21.containment (rules : Rules V Op T S) (X : N → Set T) (n : N) (t : Tree N V Op)
    (hn : n ∉ reads t) (U : Set T) (hU : U.Nonempty) (hX : (X n).Nonempty) :
    lifted rules (Function.update X n U) t = lifted rules X t := by
  ext s
  constructor
  · rintro ⟨cols, hcols, he⟩
    obtain ⟨x, hx⟩ := hX
    refine ⟨Function.update cols n x, fun m => ?_, ?_⟩
    · by_cases hm : m = n
      · subst hm
        simpa using hx
      · simpa [Function.update_of_ne hm] using hcols m
    · rw [← he]
      apply evaluate_congr
      intro m hm
      have : m ≠ n := fun h => hn (h ▸ hm)
      simp [Function.update_of_ne this]
  · rintro ⟨cols, hcols, he⟩
    obtain ⟨u, hu⟩ := hU
    refine ⟨Function.update cols n u, fun m => ?_, ?_⟩
    · by_cases hm : m = n
      · subst hm
        simpa using hu
      · simpa [Function.update_of_ne hm] using hcols m
    · rw [← he]
      apply evaluate_congr
      intro m hm
      have : m ≠ n := fun h => hn (h ▸ hm)
      simp [Function.update_of_ne this]

end Lifted

end Spec.Node
