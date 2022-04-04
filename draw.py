"""Code to create the draw for LSS Flesh and Blood
"""
__author__ = 'Chris Pearce'
__company__ =  'Technical Division'
__date__ = '28 March 2022'


import numpy as np
from collections import defaultdict, Counter
from random import shuffle

class Points():
    """
    Hyperparameter set for scoring pairings
    """
    valid                          =  16
    self_vs_self                   = -2048
    not_in_pod                     = -2048
    repeat_opponent_within_format  = -512
    repeat_opponent_between_format = -2
    multiple_byes                  = -2
    bye_last_round                 = -512
    overall_win_distance           = -8
    tiebreak_distance              = -4

BYE  = 0
PID  = 0
WINS = 1
CTB  = 2
POD  = 3
PODW = 4
PML  = 5
OML  = 6
OCTB = 7
RPLD = 8 # No of rounds played (ex byes)

def create_draw(players, results, rounds, mode='production', pod_size=8, minimiseTiebreakDist=False, seed=231):
    """
    Create draw for the round
    Parameters
    ----------
    players: Array of player details
        (n x 3) list containing [player_id, player_name, player_ranking]
    results: List of pairings and results for all rounds to date
        (m x 5) list containing [player_a_id, player_b_id, winner_id, round_id, pod_id]
    rounds: List of round types
        (n x 0) list of {'swissx' | 'podx'}
    mode: string
        System run mode. Returns reduced form outputs in production mode
    pod_size: int
        size of pods for the system to create if round is the start of a new pod
    minimiseTiebreakDist: Flag if final round. Pair scores will include distance between tiebreakers
        boolean
    Returns
    -------
    standings: List of standings for all players still in the tournament
        (n x 2) list containing [player_id, player_wins]
    splits: List of index points where each points bracket finishes
    scores: Matrix containing the scores for all possible pairings
        (n x n) matrix of floats
    alloc: The matrix form pairing allocations
        (n x n) matrix of integers
    pairs: The list form of pairing allocations
        (n / 2 x (2, 2, 1)) list containing [(player_a_standing, player_a_id), (player_b_standing, player_b_id), pod]
    """
    if seed:
        np.random.seed(seed=seed)
    
    n_players = len(players)
    results = [] if results == None else results

    standings               = get_standings(players, results, rounds, pod_size)
    scores                  = get_scores(standings, results, rounds, minimiseTiebreakDist)
    alloc, pairs, splits    = get_allocations(standings, scores, mode)

    return standings, splits, scores, alloc, pairs

def get_standings(players, results, rounds, pod_size=8, this_round=None, minimiseTiebreakDist=False):
    """
    Calculate player standings at start of round
    Parameters
    ----------
    players: Array of player details
        (n x 3) list containing [player_id, player_name, player_ranking]
    results: List of pairings and results for all rounds to date
        (m x 5) list containing [player_a_id, player_b_id, winner_id, round_id, pod_id]
    rounds: List of round types
        (n x 0) list of {'swissx' | 'podx'}
    pod_size: int
        Number of players per pod
    this_round: int
        Set the current round that standings are being calculated for
        If "None" then it will calculate for the maximum round contained in "results" plus 1
    Returns
    -------
    standings: List of standings for all players still in the tournament
        (n x 3) list containing [player_id, player_wins, player_pod]
    """
    results = [] if results == None else results
    if this_round == None:
        this_round  = 0 if len(results) == 0 else np.array(results)[:,3].astype(int).max() + 1
    last_round        = this_round - 1
    win_history       = defaultdict(lambda: np.zeros(max(1, this_round), dtype=int))
    loss_history      = defaultdict(lambda: np.zeros(max(1, this_round), dtype=int))
    pod_standings     = defaultdict(int)
    pods              = defaultdict(lambda: 1)
    active_players    = set([player[0] for player in players])
    id_to_pos         = dict()

    # Use all rounds for pod standing if 'swiss', or just pod results if 'pod'
    same_format = set([n for n, rnd in enumerate(rounds) if rnd == rounds[this_round]])

    # Calculate player standings
    if len(results) > 0:
        for a, b, winner, rnd, pod in results:
            win_history[a][rnd]  += (a == winner)
            win_history[b][rnd]  += (b == winner)
            loss_history[a][rnd] += (b == winner)
            loss_history[b][rnd] += (a == winner)
            if rnd in same_format:
                pod_standings[a] += (a == winner)
                pod_standings[b] += (b == winner)
            if rnd == last_round:
                if a in active_players: pods[a] = pod
                if b in active_players: pods[b] = pod

    # Calculate tiebreakers
    w = 0.25 ** np.arange(max(1, this_round))
    w = w.cumsum()[::-1]

    standings = np.zeros((len(active_players), 9), dtype=object)
    for n, p in enumerate(active_players):
        id_to_pos[p] = n
        standings[n,PID]  = p
        standings[n,WINS] = win_history[p].sum()
        # Calculate cumulative tiebreakers (CTB)
        standings[n,CTB] = (w*win_history[p]).sum() / w.sum()
        # Calculate player match loss (PML)
        standings[n,PML] = (loss_history[p]).mean()

    # Calculate opponent match loss (OML) and opponent cumulative tiebreak (OCTB)
    for result in results:
        if (result[0][:3] != 'BYE') & (result[1][:3] != 'BYE'):
            player1, player2 = id_to_pos[result[0]], id_to_pos[result[1]]
            standings[player1, OML]  += standings[player2, PML]
            standings[player2, OML]  += standings[player1, PML]
            standings[player1, OCTB] += standings[player2, CTB]
            standings[player2, OCTB] += standings[player1, CTB]
            standings[player1, RPLD] += 1
            standings[player2, RPLD] += 1

    for standing in standings:
        standing[OML]  /= max(standing[RPLD], 1e-12)
        standing[OCTB] /= max(standing[RPLD], 1e-12)

    standings = sorted(standings, key=lambda x: [-x[WINS], -x[CTB], x[PML], x[OML], -x[OCTB]])
    standings = np.array(standings)

    # Determine whether to create, reuse or ignore pods
    if rounds[this_round][:3] == 'pod':
        if this_round == 0 or rounds[this_round] != rounds[last_round]:
            method = 'create pod'
        else:
            method = 'reuse pod'
    else:
        method = 'bracket'

    # Allocate players to pods
    for pos, standing in enumerate(standings):
        player_id = standing[PID]
        if method == 'create pod':
            if pos < (len(players) // pod_size) * pod_size:
                standing[POD] = pos // pod_size + 1
            elif (len(players) % pod_size) / pod_size < 0.5:
                standing[POD] = pos // pod_size
            else:
                standing[POD] = pos // pod_size + 1
        elif method == 'reuse pod':
            standing[POD] = pods[player_id]
        else:
            standing[POD] = 1

    # Add byes into the largest bracket in each of the pods
    pod_sizes = Counter(standings[:,POD])
    byes = []
    for pod_num in range(1, len(pod_sizes) + 1):
        if pod_sizes[pod_num] % 2 == 1:
            min_overall_ranking  = standings[standings[:,POD]==pod_num][:,WINS].min()
            min_pod_ranking      = standings[standings[:,POD]==pod_num][:,PODW].min()
            min_pod_tiebreak     = standings[standings[:,POD]==pod_num][:,CTB].min()
            byes.append(['BYE' + str(pod_num), min_overall_ranking-1, min_pod_tiebreak, pod_num, min_pod_ranking-1, 1, 1, 0, 0])

    if len(byes) > 0:
        byes = np.array(byes, dtype=object)
        standings = np.vstack([standings, byes])

    standings = sorted(standings, key=lambda x: [x[POD], -x[WINS], -x[CTB], x[PML], x[OML], -x[OCTB]])
    standings = np.array(standings)

    return standings

def get_scores(standings, results, rounds, minimiseTiebreakDist=False):
    """
    Assign scores to each possible pairing
    A higher score means that the pairing is more strongly preferred
    Parameters
    ----------
    standings: List of standings for all players still in the tournament
        (n x 2) list containing [player_id, player_wins]
    results: List of pairings and results for all rounds to date
        (m x 5) list containing [player_a_id, player_b_id, winner_id, round_id, pod_id]
    rounds: List of round types
        (n x 0) list of {'swissx' | 'podx'}
    minimiseTiebreakDist: Flag if final round. Pair scores will include distance between tiebreakers
        boolean
    Returns
    -------
    scores: Matrix containing the scores for all possible pairings
        (n x n) matrix of floats
    """
    results = [] if results == None else results
    
    # Get player data
    n_players = len(standings)
    ix = {player_id: pos for pos, player_id in enumerate(standings[:,PID])}
    byes_ix = [n for n, standing in enumerate(standings[:,PID]) if standing[:3] == 'BYE']
    active_players = set(ix.keys())

    # Get round data
    this_round  = 0 if len(results) == 0 else np.array(results)[:,3].astype(int).max() + 1
    same_format = set([n for n, rnd in enumerate(rounds) if rnd == rounds[this_round]])

    # Calculate scores for pairings
    scores = np.full((n_players, n_players), Points.valid)

    # Set self vs self to -inf
    np.fill_diagonal(scores, Points.self_vs_self)

    # Set pairs in different pod to -inf
    not_in_pod = standings[:,POD:POD+1] != standings[:,POD:POD+1].T
    scores += Points.not_in_pod * not_in_pod

    def apply_penalty(scores, ix_player_a, ix_player_b, penalty):
        scores[ix_player_a, ix_player_b] += penalty
        scores[ix_player_b, ix_player_a] += penalty

    # Subtract points for prior matchup
    for player_a, player_b, winner, rnd, pod_id in results:
        if player_a in active_players and player_b in active_players:
            if   player_a[:3] != 'BYE' and player_b[:3] != 'BYE' and rnd in same_format:
                apply_penalty(scores, ix[player_a], ix[player_b], Points.repeat_opponent_within_format)
            elif player_a[:3] != 'BYE' and player_b[:3] != 'BYE' and rnd not in same_format and not minimiseTiebreakDist:
                apply_penalty(scores, ix[player_a], ix[player_b], Points.repeat_opponent_between_format)

    # Subtract points for repeat bye
    for player_a, player_b, winner, rnd, pod_id in results:
        penalty = Points.bye_last_round if rnd == this_round - 1 else Points.multiple_byes
        if player_a in active_players and player_b[:3] == 'BYE':
            apply_penalty(scores, ix[player_a], byes_ix, penalty)
        if player_b in active_players and player_a[:3] == 'BYE':
            apply_penalty(scores, byes_ix, ix[player_b], penalty)

    # Subtract points for different number of overall wins
    overall_win_distance = (standings[:,[WINS]] - standings[:,[WINS]].T) ** 2
    scores = scores + Points.overall_win_distance * overall_win_distance

    # Subtract points for different number of in-format wins
    if minimiseTiebreakDist:
        tiebreak_distance = 1 + 1e-4 - (np.abs(standings[:,[CTB]] - standings[:,[CTB]].T).astype(np.float32))
        scores += Points.tiebreak_distance  * -np.log(tiebreak_distance) * 25

    return scores

def get_allocations(standings, scores, mode):
    """
    Split table down into even length brackets and sub-allocate players to pairs within the bracket
    Parameters
    ----------
    standings: List of standings for all players still in the tournament
        (n x 4) list containing [player_id, player_wins, pod_id, player_wins_in_pod]
    scores: Matrix containing the scores for all possible pairings
        (n x n) matrix of floats
    Returns
    -------
    alloc: The matrix form pairing allocations
        (n x n) matrix of integers
    pairs: The list form of pairing allocations
        (n / 2 x (2, 2, 1)) list containing [(player_a_standing, player_a_id), (player_b_standing, player_b_id), pod]
    """
    n_players = len(standings)
    splits = get_splits(standings)

    brackets = [n for n in range(n_players) if (standings[n,[WINS]] != standings[n-1,[WINS]]).any()] + [n_players]
    n_odd_brackets = sum(np.array(brackets) % 2)
    max_score = Points.valid  - 2 * n_odd_brackets * (Points.valid + Points.overall_win_distance) / n_players

    # Run Gibbs Sampler, and return if it hit max result
    allocs_gs, pairs_gs = _gibbs_sampler(standings, scores)
    if scores[allocs_gs==1].mean(dtype=float) == max_score:
        return allocs_gs, pairs_gs, splits

    # Otherwise run Metropolis-Hastings Sampler
    # Allocate entire table if there is no split
    if len(splits) == 1:
        allocs, pairs = _metroplis_hastings(standings, scores)

    else:
        allocs = np.zeros((n_players, n_players))
        pairs = []

        # Split into sub-tables at even numbered index split points
        ixs = np.array(splits)
        ixs = ixs[ixs % 2 == 0]
        for ix1, ix2 in zip(ixs[:-1], ixs[1:]):
            alloc, pair = _metroplis_hastings(standings[ix1:ix2], scores[ix1:ix2,ix1:ix2], start=ix1)
            allocs[ix1:ix2,ix1:ix2] = alloc
            pairs += pair

    allocs_mh, pairs_mh = _gibbs_sampler(standings, scores, alloc=allocs)

    # Return the better of Gibbs or Metropolis Hastings
    if(scores[allocs_mh==1].mean(dtype=float) >= scores[allocs_gs==1].mean(dtype=float)):
        return allocs_mh, pairs_mh, splits
    else:
        return allocs_gs, pairs_gs, splits

def _gibbs_sampler(standings, scores, start=0, alloc=None):
    """
    Allocate player pairings using Gibbs' sampling method
    Parameters
    ----------
    standings: List of standings for all players still in the tournament
        (n x 4) list containing [player_id, player_wins, pod_id, player_wins_in_pod]
    scores: Matrix containing the scores for all possible pairings
        (n x n) matrix of floats
    Returns
    -------
    alloc: The matrix form pairing allocations
        (n x n) matrix of integers
    pairs: The list form of pairing allocations
        (n / 2 x (2, 2, 1)) list containing [(player_a_standing, player_a_id), (player_b_standing, player_b_id), pod]
    """
    n_players = len(standings)
    ix = {pos+start: player_id for pos, player_id in enumerate(standings[:,PID])}
    pods = {player[PID]: player[POD] for player in standings}

    # Construct initial draw
    if alloc is None:
        alloc = np.zeros((n_players, n_players), dtype=int)
        for _ in range(int(n_players / 2)):
            mask = (1 - alloc.sum(axis=1))
            # Choose Player A
            a = mask.argmax()
            # Choose Player B
            p = scores[a][mask == 1]
            p = p - p.max()
            p = np.exp(p.astype(np.float32)/4)
            p = p / p.sum()
            b = np.random.choice(np.arange(n_players)[mask == 1], p=p)

            alloc[a, b] = 1
            alloc[b, a] = 1

    # Run one-up-one-down to iteratively improve draw
    score_q = [(alloc*scores).sum()/n_players]

    brackets = [n for n in range(n_players) if (standings[n,[WINS]] != standings[n-1,[WINS]]).any()] + [n_players]
    n_odd_brackets = sum(np.array(brackets) % 2)
    max_score = Points.valid  - 2 * n_odd_brackets * (Points.valid + Points.overall_win_distance) / n_players

    while max(score_q) < max_score:
        # If the worst score is in the first half of the draw,  run through first to last
        # If the worst score is in the second half of the draw, run through last to first
        isFwdSort = scores[alloc==1].argmin() / len(scores) < 0.5
        isFwdSort = (-1)**(1+isFwdSort)

        for a in range(n_players-1)[::isFwdSort]:
            b = alloc[a].argmax()
            if scores[a,b] == scores[a].max():
                continue

            # Search one-up-one-down
            # Creates series c = [a+1, a-1, a+2, a-2...]
            for i in range(2, 2 * n_players):
                # Select c and d
                c = a + (i // 2) * (-1)**i
                if c not in range(n_players-1):
                    continue
                d = alloc[c].argmax()

                opts = [scores[a,b]+scores[c,d],
                        scores[a,c]+scores[b,d],
                        scores[a,d]+scores[b,c]]
                # Swap pairs if there is a better allocation
                if max(opts[1], opts[2]) > opts[0]:
                    opt = np.argmax(opts)

                    if opt == 1: alloc = swap_pairs((a,b), (c,d), alloc)
                    if opt == 2: alloc = swap_pairs((a,b), (d,c), alloc)
                    break

        score = (alloc*scores).sum()/n_players
        score_q.append(score)
        if score_q[-1] == score_q[-2]:
            break

    #if len(score_q) > 1: print(["%.2f" % v for v in score_q])

    # Create list form of pairings
    pairs = []
    for a in range(n_players):
        b = alloc[a].argmax()
        if a < b:
            a, b = a + start, b + start
            pairs.append([(a, ix[a]), (b, ix[b]), pods[ix[a]]])

    return alloc, pairs

def _metroplis_hastings(standings, scores, start=0):
    """
    Allocate player pairings using MH probability method
    Parameters
    ----------
    standings: List of standings for all players still in the tournament
        (n x 4) list containing [player_id, player_wins, pod_id, player_wins_in_pod]
    scores: Matrix containing the scores for all possible pairings
        (n x n) matrix of floats
    Returns
    -------
    alloc: The matrix form pairing allocations
        (n x n) matrix of integers
    pairs: The list form of pairing allocations
        (n / 2 x (2, 2, 1)) list containing [(player_a_standing, player_a_id), (player_b_standing, player_b_id), pod]
    """
    n_players = len(standings)
    ix = {pos+start: player_id for pos, player_id in enumerate(standings[:,PID])}
    alloc = np.zeros((n_players, n_players), dtype=int)
    pods = {player[PID]: player[POD] for player in standings}

    # Construct initial draw
    constraints = scores.astype(int) == scores.astype(int).max(axis=1, keepdims=True)
    constraints = n_players - constraints.sum(axis=1)
    prob = scores - scores.max(axis=1, keepdims=True)
    prob = np.exp(prob.astype(np.float32)/4)

    for _ in range(int(n_players / 2)):
        mask = (1 - alloc.sum(axis=1))
        # Choose Player A
        a = (constraints * mask).argmax()
        # Choose Player B
        p = scores[a][mask == 1]
        p = p - p.max()
        p = np.exp(p.astype(np.float32)/4)
        p = p / p.sum()
        b = np.random.choice(np.arange(n_players)[mask == 1], p=p)

        alloc[a, b] = 1
        alloc[b, a] = 1

    brackets = [n for n in range(n_players) if (standings[n,[WINS]] != standings[n-1,[WINS]]).any()] + [n_players]
    n_odd_brackets = sum(np.array(brackets) % 2)
    max_score = Points.valid  - 2 * n_odd_brackets * (Points.valid + Points.overall_win_distance) / n_players

    # Run Metropolis-Hastings to iteratively improve draw
    score_q = [(alloc*scores).sum()/len(scores)]
    if score_q[0] != max_score:
        prob /= prob.sum(axis=1, keepdims=True)

        best_alloc = alloc.copy()
        i = 0
        while len(score_q) < 100 or np.mean(score_q[-10:]) > np.mean(score_q[-20:-10]):
            i += 1
            if max(score_q) == max_score:
                break
            if len(score_q) > 10 and min(score_q[-10:]) == max(score_q[-10:]):
                break
            for a, p in enumerate(prob):
                b = alloc[a].argmax()
                c = np.random.choice(range(len(p)),p=p)
                d = alloc[c].argmax()
                p_change = prob[a,c]*prob[b,d]/(prob[a,b]*prob[c,d] + prob[a,c]*prob[b,d])
                if np.random.random() < p_change:
                    alloc = swap_pairs((a,b), (c,d), alloc)

            score = (alloc*scores).sum()/len(prob)
            if score > max(score_q):
                best_alloc = alloc.copy()
            score_q.append(score)

        alloc = best_alloc
        del(best_alloc)

    # Create list form of pairings
    alloc_ = alloc * np.tri(*alloc.shape)

    pairs = []
    for n in range(n_players):
        if alloc_[n].max() == 1:
            a, b = n + start, alloc_[n].argmax() + start
            pairs.append([(a, ix[a]), (b, ix[b]), pods[ix[a]]])

    return alloc, pairs

def swap_pairs(pair_1, pair_2, alloc):
    """
    Helper function to swap players between two pairings
    Original pairing of (a,b) and (c,d) becomes (a,c) and (b,d)
    Parameters
    ----------
    pair_1: Tuple containing first pair to swap (a,b)
    pair_2: Tuple containing first pair to swap (c,d)
    alloc: The matrix form pairing allocations
        (n x n) matrix of integers
    Returns
    -------
    alloc: The matrix form pairing allocations with [(a,b), (c,d)] -> [(a,c), (b,d)]
        (n x n) matrix of integers
    """
    a, b = pair_1
    c, d = pair_2

    alloc[a, b] = 0
    alloc[b, a] = 0
    alloc[c, d] = 0
    alloc[d, c] = 0
    alloc[a, c] = 1
    alloc[c, a] = 1
    alloc[b, d] = 1
    alloc[d, b] = 1

    return alloc

def get_splits(standings):
    """
    Helper method to find the points at which the brackets can be split with even numbers of players
    Parameters
    ----------
    standings: List of standings for all players still in the tournament
        (n x 4) list containing [player_id, player_wins, pod_id, player_wins_in_pod]
    Returns
    -------
    splits: List of indices where brackets can be split
        (b x 0) list of integers [breakpoint1, breakpoint2, ..., breakpointb]
    """
    n_players = len(standings)
    n_pods    = len(set(standings[:,POD]))

    # Sort standings by bracket and player wins
    standings = sort(standings, [CTB, WINS, POD], [1, -1, 1])

    # Calculate splits
    if n_pods > 1:
        # If players are allocated to pods
        # Add split if there is a break across pods
        splits = [n for n in range(n_players) if (standings[n,[POD]] != standings[n-1,[POD]]).any()] + [n_players]
    else:
        # If players are not allocated to pods
        # Add split if more than 40 players are on the same overall points
        splits, breakpoints = [0], [0]
        breakpoints += [n for n in range(1, n_players) if (standings[n,[WINS]] != standings[n-1,[WINS]]).any()] + [n_players]
        for break_1, break_2 in zip(breakpoints[:-1], breakpoints[1:]):
            if break_2 - break_1 > 40:
                splits += list(range(break_1 + 40 + break_1 % 2, break_2 - 40, 40))
        splits.append(n_players)

    return splits

def sort(x, cols, orders, preshuffle=False):
    """
    Helper method to sort an array by a combination of columns in forward or reverse orders
    Parameters
    ----------
    x: Array to sort
        (n x m) numpy array
    cols: List of the columns to sort by
        list [(0-m), ..., (0-m)]
    orders: List of the order to to sort columns by. 1 == ascending, -1 == descending
        list [(1,-1), ..., (1,-1)]
    preshuffle: Flag whether to shuffle before sorting
        boolean
    Returns
    -------
    x: Sorted array
        (n x m) numpy array
    """
    assert(len(cols) == len(orders))

    if preshuffle:
        x = x.copy()
        np.random.shuffle(x)

    for col, order in zip(cols, orders):
        ind = np.argsort(x[:,col], kind='stable')
        x   = x[ind][::order]

    return x