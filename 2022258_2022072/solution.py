"""
Nash Equilibrium Solver for Finite n-Player Strategic Form Games
Uses Z3 SMT Solver for both Pure and Mixed Strategy Nash Equilibria.

Usage:
    python nash_equilibrium.py

The game is specified by:
    - Number of players
    - Number of actions per player
    - Payoff tensors for each player
"""

from z3 import *
from itertools import product


# ============================================================
# SECTION 1: Game Representation
# ============================================================

class Game:
    """
    Represents a finite n-player strategic form game.

    Attributes:
        num_players (int): Number of players.
        num_actions (list[int]): Number of actions available to each player.
        payoffs (dict): Mapping from action profile (tuple) to list of payoffs.
                        payoffs[(a0, a1, ..., a_{n-1})][i] = payoff to player i.
    """

    def __init__(self, num_players, num_actions, payoffs):
        self.num_players = num_players
        self.num_actions = num_actions  # list: num_actions[i] = number of actions for player i
        self.payoffs = payoffs          # dict: (a0, a1, ..., a_{n-1}) -> [payoff_0, ..., payoff_{n-1}]

    def action_profiles(self):
        """Returns an iterator over all action profiles (Cartesian product of action sets)."""
        return product(*[range(self.num_actions[i]) for i in range(self.num_players)])

    def payoff(self, profile, player):
        """Returns the payoff to 'player' under the given action profile (tuple)."""
        return self.payoffs[tuple(profile)][player]


# ============================================================
# SECTION 2: Pure Strategy Nash Equilibrium (Brute Force)
# ============================================================

def find_pure_nash(game):
    """
    Finds all Pure Strategy Nash Equilibria by checking every action profile.

    Algorithm:
        For each action profile (a_0, ..., a_{n-1}):
            For each player i:
                Check if player i can improve their payoff by unilaterally
                deviating to any other action a'_i.
            If no player can improve -> it is a PSNE.

    Returns:
        List of action profiles (tuples) that are pure strategy Nash equilibria.
    """
    equilibria = []

    for profile in game.action_profiles():
        profile = list(profile)
        is_nash = True

        for player in range(game.num_players):
            current_payoff = game.payoff(profile, player)
            # Check all deviations for this player
            for alt_action in range(game.num_actions[player]):
                if alt_action == profile[player]:
                    continue
                deviated_profile = profile.copy()
                deviated_profile[player] = alt_action
                if game.payoff(deviated_profile, player) > current_payoff:
                    is_nash = False
                    break
            if not is_nash:
                break

        if is_nash:
            equilibria.append(tuple(profile))

    return equilibria


# ============================================================
# SECTION 3: Mixed Strategy Nash Equilibrium (Z3 SMT Solver)
# ============================================================

def find_mixed_nash(game):
    """
    Finds a Mixed Strategy Nash Equilibrium using Z3 SMT Solver.

    Algorithm (Indifference / Support Enumeration via SMT):
        1. For each player i, define probability variables p_i_j for each action j.
        2. Add constraints:
           a) p_i_j >= 0 for all i, j                  (non-negativity)
           b) sum_j p_i_j = 1 for all i                (probability simplex)
        3. For each player i and each action j:
           Compute EU_i(j) = expected utility of player i playing action j,
           given the mixed strategies of all other players.
        4. Nash Equilibrium condition (for each player i):
           For every pair of actions j, k of player i:
               If p_i_j > 0 then EU_i(j) >= EU_i(k)
           This ensures that every action in the support yields the
           maximum expected utility.
        5. Optionally, to find a non-trivial (non-pure) MSNE, we can add
           a constraint that at least one player mixes (has support size > 1).
        6. Solve using Z3.

    Returns:
        Dictionary mapping player index to list of probabilities, or None.
    """
    solver = Solver()

    # --- Step 1: Define probability variables ---
    # prob[i][j] = probability that player i plays action j
    prob = []
    for i in range(game.num_players):
        player_probs = []
        for j in range(game.num_actions[i]):
            p = Real(f'p_{i}_{j}')
            player_probs.append(p)
        prob.append(player_probs)

    # --- Step 2: Probability constraints ---
    for i in range(game.num_players):
        # Non-negativity
        for j in range(game.num_actions[i]):
            solver.add(prob[i][j] >= 0)
        # Sum to 1
        solver.add(Sum(prob[i]) == 1)

    # --- Step 3: Compute expected utilities ---
    # EU[i][j] = expected utility of player i when player i plays action j
    #            and all other players play according to their mixed strategies.
    EU = []
    for i in range(game.num_players):
        eu_player = []
        for j in range(game.num_actions[i]):
            # Sum over all action profiles of other players
            other_players = [p for p in range(game.num_players) if p != i]
            other_action_ranges = [range(game.num_actions[p]) for p in other_players]

            expected_util = RealVal(0)
            for other_actions in product(*other_action_ranges):
                # Build the full action profile
                profile = [0] * game.num_players
                profile[i] = j
                for idx, p in enumerate(other_players):
                    profile[p] = other_actions[idx]

                # Payoff for this profile
                payoff_val = game.payoff(profile, i)

                # Probability of other players playing this combination
                prob_others = RealVal(1)
                for idx, p in enumerate(other_players):
                    prob_others = prob_others * prob[p][other_actions[idx]]

                expected_util = expected_util + RealVal(payoff_val) * prob_others

            eu_player.append(expected_util)
        EU.append(eu_player)

    # --- Step 4: Nash equilibrium constraints ---
    # For each player i, for each action j:
    #   If p_i_j > 0, then EU_i(j) >= EU_i(k) for all k.
    # Equivalently: p_i_j > 0 => EU_i(j) >= max_k EU_i(k)
    # Which means: for all j, k: p_i_j > 0 => EU_i(j) >= EU_i(k)
    for i in range(game.num_players):
        for j in range(game.num_actions[i]):
            for k in range(game.num_actions[i]):
                if j != k:
                    solver.add(Implies(prob[i][j] > 0, EU[i][j] >= EU[i][k]))

    # --- Step 5: Try to find a mixed (non-pure) equilibrium first ---
    # Add constraint: at least one player has support size >= 2
    mixing_constraints = []
    for i in range(game.num_players):
        # Player i mixes if at least 2 actions have positive probability
        for j in range(game.num_actions[i]):
            for k in range(j + 1, game.num_actions[i]):
                mixing_constraints.append(And(prob[i][j] > 0, prob[i][k] > 0))

    # First try: find a non-pure mixed NE
    solver.push()
    solver.add(Or(mixing_constraints))

    result = solver.check()
    if result == sat:
        model = solver.model()
        strategy = extract_strategy(model, prob, game)
        return strategy, False  # False = not purely pure

    # If no non-pure mixed NE found, try without the mixing constraint
    solver.pop()
    result = solver.check()
    if result == sat:
        model = solver.model()
        strategy = extract_strategy(model, prob, game)
        return strategy, True  # True = it's actually a pure NE in mixed form

    return None, None


def extract_strategy(model, prob, game):
    """Extracts the mixed strategy from a Z3 model."""
    strategy = {}
    for i in range(game.num_players):
        probs = []
        for j in range(game.num_actions[i]):
            val = model.evaluate(prob[i][j])
            # Convert Z3 rational to float
            if is_rational_value(val):
                probs.append((val.numerator_as_long(), val.denominator_as_long()))
            else:
                probs.append((float(val.as_decimal(10).rstrip('?')), 1))
        strategy[i] = probs
    return strategy


def format_fraction(num, den):
    """Format a fraction nicely."""
    if den == 1:
        return str(num)
    else:
        return f"{num}/{den}"


def format_probability(prob_tuple):
    """Format probability tuple (numerator, denominator) as string."""
    if isinstance(prob_tuple[1], int) or (isinstance(prob_tuple[1], float) and prob_tuple[1] == 1.0):
        num, den = prob_tuple
        if isinstance(num, int) and isinstance(den, int):
            return format_fraction(num, den)
        else:
            return f"{num}"
    return f"{prob_tuple[0]}/{prob_tuple[1]}"


# ============================================================
# SECTION 4: Example Games and Main Driver
# ============================================================

def create_prisoners_dilemma():
    """Classic 2-player Prisoner's Dilemma."""
    # Actions: 0 = Cooperate, 1 = Defect
    payoffs = {
        (0, 0): [-1, -1],    # Both cooperate
        (0, 1): [-3, 0],     # P1 cooperates, P2 defects
        (1, 0): [0, -3],     # P1 defects, P2 cooperates
        (1, 1): [-2, -2],    # Both defect
    }
    return Game(2, [2, 2], payoffs)


def create_battle_of_sexes():
    """Classic 2-player Battle of the Sexes."""
    # Actions: 0 = Opera, 1 = Football
    payoffs = {
        (0, 0): [3, 2],
        (0, 1): [0, 0],
        (1, 0): [0, 0],
        (1, 1): [2, 3],
    }
    return Game(2, [2, 2], payoffs)


def create_matching_pennies():
    """Classic 2-player Matching Pennies (no pure NE)."""
    # Actions: 0 = Heads, 1 = Tails
    payoffs = {
        (0, 0): [1, -1],
        (0, 1): [-1, 1],
        (1, 0): [-1, 1],
        (1, 1): [1, -1],
    }
    return Game(2, [2, 2], payoffs)


def create_rock_paper_scissors():
    """Classic 2-player Rock-Paper-Scissors (no pure NE)."""
    # Actions: 0=Rock, 1=Paper, 2=Scissors
    payoffs = {
        (0, 0): [0, 0], (0, 1): [-1, 1], (0, 2): [1, -1],
        (1, 0): [1, -1], (1, 1): [0, 0], (1, 2): [-1, 1],
        (2, 0): [-1, 1], (2, 1): [1, -1], (2, 2): [0, 0],
    }
    return Game(2, [3, 3], payoffs)


def create_three_player_game():
    """A 3-player game with 2 actions each."""
    # Player 0: actions {0, 1}
    # Player 1: actions {0, 1}
    # Player 2: actions {0, 1}
    payoffs = {
        (0, 0, 0): [2, 2, 2],
        (0, 0, 1): [0, 1, 1],
        (0, 1, 0): [1, 0, 1],
        (0, 1, 1): [1, 1, 0],
        (1, 0, 0): [1, 1, 0],
        (1, 0, 1): [1, 0, 1],
        (1, 1, 0): [0, 1, 1],
        (1, 1, 1): [3, 3, 3],
    }
    return Game(3, [2, 2, 2], payoffs)


def create_custom_game():
    """Allows user to input a custom game interactively."""
    print("\n=== Custom Game Input ===")
    n = int(input("Enter number of players: "))
    num_actions = []
    for i in range(n):
        a = int(input(f"Enter number of actions for Player {i}: "))
        num_actions.append(a)

    payoffs = {}
    print(f"\nEnter payoffs for each action profile.")
    print(f"Each profile is a tuple of actions, one per player.")
    print(f"For each profile, enter {n} payoff values separated by spaces.\n")

    for profile in product(*[range(num_actions[i]) for i in range(n)]):
        payoff_str = input(f"  Payoffs for profile {profile}: ")
        payoff_vals = list(map(float, payoff_str.strip().split()))
        assert len(payoff_vals) == n, f"Expected {n} values, got {len(payoff_vals)}"
        payoffs[profile] = payoff_vals

    return Game(n, num_actions, payoffs)


def print_separator():
    print("=" * 60)


def solve_game(game, game_name="Game", action_names=None):
    """
    Solves a game for both pure and mixed strategy Nash equilibria.
    
    Args:
        game: Game object
        game_name: Name for display
        action_names: Optional dict mapping player_index -> list of action names
    """
    print_separator()
    print(f"  SOLVING: {game_name}")
    print(f"  Players: {game.num_players}, Actions: {game.num_actions}")
    print_separator()

    # --- Part (a): Pure Strategy Nash Equilibrium ---
    print("\n--- Part (a): Pure Strategy Nash Equilibria ---\n")
    pure_equilibria = find_pure_nash(game)

    if pure_equilibria:
        print(f"  Found {len(pure_equilibria)} Pure Strategy Nash Equilibri(um/a):\n")
        for idx, eq in enumerate(pure_equilibria):
            if action_names:
                named = tuple(action_names[i][eq[i]] for i in range(game.num_players))
                print(f"    PSNE {idx + 1}: {eq}  =>  {named}")
            else:
                print(f"    PSNE {idx + 1}: {eq}")
            # Print payoffs at this equilibrium
            payoff_at_eq = game.payoffs[eq]
            print(f"             Payoffs: {payoff_at_eq}")
    else:
        print("  No Pure Strategy Nash Equilibrium exists.\n")

    # --- Part (b): Mixed Strategy Nash Equilibrium ---
    print("\n--- Part (b): Mixed Strategy Nash Equilibrium ---\n")
    mixed_result, is_pure = find_mixed_nash(game)

    if mixed_result is not None:
        if is_pure:
            print("  (Note: The mixed NE found is actually a pure strategy NE)\n")
        print("  Mixed Strategy Nash Equilibrium found:\n")
        for i in range(game.num_players):
            probs = mixed_result[i]
            prob_strs = []
            for j in range(game.num_actions[i]):
                p_str = format_probability(probs[j])
                if action_names:
                    prob_strs.append(f"{action_names[i][j]}: {p_str}")
                else:
                    prob_strs.append(f"Action {j}: {p_str}")
            print(f"    Player {i}: [{', '.join(prob_strs)}]")

        # Compute expected payoffs under the mixed NE
        print("\n  Expected payoffs under this MSNE:")
        for i in range(game.num_players):
            exp_payoff = 0.0
            for profile in game.action_profiles():
                profile = tuple(profile)
                prob_profile = 1.0
                for p in range(game.num_players):
                    num, den = mixed_result[p][profile[p]]
                    prob_profile *= num / den if isinstance(den, int) and den != 0 else num
                exp_payoff += prob_profile * game.payoff(profile, i)
            print(f"    Player {i}: {exp_payoff:.6f}")
    else:
        print("  Z3 could not find a Mixed Strategy Nash Equilibrium.")
        print("  (This should not happen for finite games — Nash's theorem guarantees existence.)")

    print()


# ============================================================
# MAIN
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("   NASH EQUILIBRIUM SOLVER")
    print("   For Finite n-Player Strategic Form Games")
    print("   Using Z3 SMT Solver")
    print("=" * 60)

    # --- Game 1: Prisoner's Dilemma ---
    game1 = create_prisoners_dilemma()
    action_names1 = {0: ["Cooperate", "Defect"], 1: ["Cooperate", "Defect"]}
    solve_game(game1, "Prisoner's Dilemma", action_names1)

    # --- Game 2: Battle of the Sexes ---
    game2 = create_battle_of_sexes()
    action_names2 = {0: ["Opera", "Football"], 1: ["Opera", "Football"]}
    solve_game(game2, "Battle of the Sexes", action_names2)

    # --- Game 3: Matching Pennies ---
    game3 = create_matching_pennies()
    action_names3 = {0: ["Heads", "Tails"], 1: ["Heads", "Tails"]}
    solve_game(game3, "Matching Pennies", action_names3)

    # --- Game 4: Rock-Paper-Scissors ---
    game4 = create_rock_paper_scissors()
    action_names4 = {0: ["Rock", "Paper", "Scissors"], 1: ["Rock", "Paper", "Scissors"]}
    solve_game(game4, "Rock-Paper-Scissors", action_names4)

    # --- Game 5: 3-Player Game ---
    game5 = create_three_player_game()
    solve_game(game5, "3-Player Game (2 actions each)")

    # --- Custom Game ---
    print("\nWould you like to input a custom game? (y/n): ", end="")
    choice = input().strip().lower()
    if choice == 'y':
        custom_game = create_custom_game()
        solve_game(custom_game, "Custom Game")


if __name__ == "__main__":
    main()