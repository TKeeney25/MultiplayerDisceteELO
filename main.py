import uuid

from copiedMultielo.multielo import MultiElo
import json
import numpy as np
import matplotlib.pyplot as plt
import difflib
import shutil

ELO = MultiElo()
RANKING_FILE = 'rankings.json'

FIRST_PLAY_BONUS = 20
PER_GAME_BONUS = 5


class NonexistentPlayerException(Exception):
    pass

class NonexistentGameException(Exception):
    pass


def add_to_commit_log(players, game, places, performance, old_elo, new_elo):
    JSON_FILE['COMMIT_LOG'].append({
        'PLAYERS': players,
        'GAME': game,
        'PLACES': places,
        'PERFORMANCE': performance,
        'DELTA_ELO': list(new_elo - old_elo)
    })
    dump_json_data()


def add_to_redo_log(players, game, places, performance):
    JSON_FILE['REDO_LOG'].append({
        'PLAYERS': players,
        'GAME': game,
        'PLACES': places,
        'PERFORMANCE': performance
    })
    dump_json_data()


def resolve_team_match(team_placement):
    all_team_members = []
    all_team_elos = []
    all_team_participation = []
    for team in team_placement:
        team_elo = 0
        team_members = []
        team_particapation = []
        for player in team:
            team_elo += player[1]
            team_members.append(player[0])
            team_particapation.append(player[2])
        all_team_members.append(team_members)
        all_team_elos.append(team_elo)
        all_team_participation.append(team_particapation)
    old_ratings = np.array(all_team_elos) / 2
    new_ratings = ELO.get_new_ratings(old_ratings)
    ratings_diff = new_ratings - old_ratings
    new_adjusted_ratings = np.array([])
    for i_team in range(0, len(all_team_members)):
        mem_gain = []
        mem_old = []
        for member in all_team_members[i_team]:
            for i_member in team_placement[i_team]:
                if member == i_member[0]:
                    old_val = old_ratings[i_team]
                    old_ratings[i_team] = i_member[1]
                    mem_old.append(i_member[1])
                    mem_ratings = ELO.get_new_ratings(old_ratings)
                    old_ratings[i_team] = old_val
                    mem_gain.append(mem_ratings[i_team] - i_member[1])
                    break
        np_mem_gain = np.array(mem_gain)
        norm = np.linalg.norm(np_mem_gain) * (np_mem_gain[i_team] / np.abs(np_mem_gain[i_team]))
        norm2 = np.linalg.norm(all_team_participation[i_team])
        np_mem_gain = (np_mem_gain / norm) * ratings_diff[i_team]
        np_mem_gain = (np_mem_gain / norm2) * team_particapation
        np_mem_gain += np.array(mem_old)
        new_adjusted_ratings = np.concatenate((new_adjusted_ratings, np_mem_gain))
    return new_adjusted_ratings


def resolve_match(game, player_tuples):
    games = JSON_FILE['PLAYERS'][list(JSON_FILE['PLAYERS'].keys())[0]]
    if game not in games:
        closest_match = difflib.get_close_matches(game, games, n=1, cutoff=0)[0]
        response = input(f'You inputted {game}, did you mean {closest_match}? (Y/N) ').upper()
        if response == 'Y':
            game = closest_match
        else:
            response = input(f'Do you wish to create a game called {game}? (Y/N) ').upper()
            if response == 'Y':
                add_game(game)
            else:
                raise NonexistentGameException(f'{game} does not exist.')

    elo_array = np.array([])
    player_tuples.sort(key=lambda x: x[1])
    players = []
    places = []
    performance = []
    team_placement = list(np.empty(player_tuples[-1][1]))
    for item in range(0, len(team_placement)):
        team_placement[item] = []
    for arg in player_tuples:
        get_stats = get_players_rating(arg[0], game)
        players_rating = (get_stats[0], get_stats[1], arg[2])
        elo_array = np.append(elo_array, [players_rating[1]])
        players.append(players_rating[0])
        places.append(arg[1])
        performance.append(arg[2])
        team_placement[arg[1] - 1].append(players_rating)
    if len(team_placement) < len(players):
        new_ratings = resolve_team_match(team_placement)
    else:
        new_ratings = ELO.get_new_ratings(elo_array)
    for i in range(0, len(players)):
        bonus = PER_GAME_BONUS
        if get_players_rating(players[i], game) == 1000:
            bonus += FIRST_PLAY_BONUS
        set_players_rating(players[i], game, new_ratings[i] + bonus)
    add_to_commit_log(players, game, places, performance, elo_array, new_ratings)


def calculate_elo(name: str, game=None) -> float:
    num = 0
    total = 0
    if game:
        return JSON_FILE['PLAYERS'][name][game]
    for player_game in JSON_FILE['PLAYERS'][name]:
        total += JSON_FILE['PLAYERS'][name][player_game]
        num += 1
    return total / num

# TODO handle different games
def get_players_rating(name: str, game=None) -> (str, float):
    name = name.upper()
    try:
        return name, calculate_elo(name, game)
    except:
        names = list(JSON_FILE['PLAYERS'].keys())
        if len(names) != 0:
            closest_match = difflib.get_close_matches(name, names, n=1, cutoff=0)[0]
            response = input(f'You inputted {name}, did you mean {closest_match}? (Y/N) ').upper()
        else:
            response = None
        if response == 'Y':
            return closest_match, calculate_elo(closest_match, game)
        else:
            response = input(f'Options:\n0: Create New Player Called {name}\n1: Enter Different Name\n2: Return\n')
            if response == '0':
                create_player(name)
                return name, calculate_elo(name, game)
            elif response == '1':
                new_name = input('Please input the different name: ')
                return get_players_rating(new_name, game)
            else:
                raise NonexistentPlayerException(f'Player {name} is not found in the JSON file.')


def set_players_rating(name: str, game: str, elo: float):
    name = name.upper()
    JSON_FILE['PLAYERS'][name][game] = elo
    dump_json_data()


DEFAULT_SCORE = 1000.0


def create_player(name: str):
    name = name.upper()
    default_player = {
        'PONG': DEFAULT_SCORE,
        'TITS': DEFAULT_SCORE,
        'DIE': DEFAULT_SCORE,
        'BALL': DEFAULT_SCORE,
        'SNAPPA': DEFAULT_SCORE,
        'TWENTY ONE 21': DEFAULT_SCORE,
        'FLIP CUP': DEFAULT_SCORE,
        'BASEBALL': DEFAULT_SCORE
    }
    JSON_FILE['PLAYERS'][name] = default_player
    dump_json_data()


def add_game(name: str):
    name = name.upper()
    for player in JSON_FILE['PLAYERS']:
        JSON_FILE['PLAYERS'][player][name] = DEFAULT_SCORE


def dump_json_data():
    with open(file=RANKING_FILE, mode='w') as ranking_file:
        json.dump(JSON_FILE, ranking_file)


def create_rankings_file():
    with open(file=RANKING_FILE, mode='w') as ranking_file:
        default_json = {
            'PLAYERS': {

            },
            'COMMIT_LOG': [

            ],
            'REDO_LOG': [

            ]
        }
        json.dump(default_json, ranking_file)


def print_rankings_file():
    print(JSON_FILE)


def get_plot(num_players=10):
    player_and_score = []
    for player in JSON_FILE['PLAYERS']:
        player_and_score.append((player, get_players_rating(player)[1]))
    player_and_score.sort(key=lambda x: x[1], reverse=True)
    player_and_score = player_and_score[0:num_players]
    # make data:
    x = np.arange(len(player_and_score))
    x_2 = np.arange(len(player_and_score)) * 2 - 1
    y = []
    players = []
    for player in player_and_score:
        players.append(player[0])
        y.append(player[1])

    # plot
    fig, ax = plt.subplots()
    scale = 1
    fig.set_size_inches(scale * 16, scale * 9)
    ax.bar(x, y)
    plt.xlabel('Standing')
    plt.ylabel('ELO')
    ax.set_xticks(range(0, len(player_and_score)))
    ax.set_xticklabels(range(1, len(player_and_score) + 1))
    ax.bar_label(ax.containers[0], labels=players, label_type='edge')
    ax.bar_label(ax.containers[0], label_type='center')
    offscreen = 50
    ax.set_ylim((y[-1] - offscreen, y[0] + offscreen))
    ax.set_xlim((-.5, len(player_and_score) - .5))
    ax.plot(x_2, len(player_and_score) * [1000], color='r', linewidth=2)
    plt.savefig('results.pdf')


class Operations:
    def add_game(self):
        game = input('Game: ')
        i1 = input('Number of Players: ')
        players = []
        for i in range(0, int(i1)):
            player = input('Player Name: ')
            place = int(input('Place: '))
            performance = float(input('Performance: '))
            print('')
            players.append((player, place, performance))
        resolve_match(game, players)

    def undo_game(self, user_prompted=True, add_to_redo=True):
        if user_prompted:
            i1 = input('Are you sure you wish to undo? (Y/N) ').upper()
        else:
            i1 = 'Y'
        if i1 == 'Y':
            if user_prompted:
                i1 = input('Do you wish to permanently delete the game? (Y/N) ').upper()
                if i1 == 'Y':
                    add_to_redo = False
            if len(JSON_FILE['COMMIT_LOG']) == 0:
                print('There are no commits to undo.')
                return False
            last_game = JSON_FILE['COMMIT_LOG'].pop()
            for i in range(0, len(last_game['PLAYERS'])):
                player_name = last_game['PLAYERS'][i]
                game = last_game['GAME']
                player_delta = last_game['DELTA_ELO'][i]
                set_players_rating(player_name, game, get_players_rating(player_name, game)[1] - player_delta)
            if add_to_redo:
                add_to_redo_log(last_game['PLAYERS'], last_game['GAME'], last_game['PLACES'], last_game['PERFORMANCE'])
            dump_json_data()
            return True

    def redo_game(self, user_prompted=True):
        if user_prompted:
            i1 = input('Are you sure you wish to redo? (Y/N) ').upper()
        else:
            i1 = 'Y'
        if i1 == 'Y':
            if len(JSON_FILE['REDO_LOG']) == 0:
                print('There are no commits to undo.')
                return False
            last_game = JSON_FILE['REDO_LOG'].pop()
            player_tuples = []
            game = last_game['GAME']
            for i in range(0, len(last_game['PLAYERS'])):
                player_name = last_game['PLAYERS'][i]
                player_place = last_game['PLACES'][i]
                player_performance = last_game['PERFORMANCE'][i]
                player_tuples.append((player_name, player_place, player_performance))
            resolve_match(game, player_tuples)
            dump_json_data()
            return True

    def remove_game(self):
        i1 = int(input('Which game do you wish to remove? '))
        failed = False
        for undone in range(0, i1):
            failed = not self.undo_game(False) or failed
        if not failed:
            self.undo_game(False, False)
        else:
            print('Failed to remove game.')
        while self.redo_game(False):
            pass

    def create_player(self):
        i1 = input('Please Enter a Valid Player Name: ')
        create_player(i1)

    def replay_all_games(self):
        while self.undo_game(False):
            pass
        while self.redo_game(False):
            pass

    def get_plot(self, user_prompted=True):
        if user_prompted:
            num = int(input('How many people do you want to portray? '))
            get_plot(num)
        else:
            get_plot()

    def create_backup(self):
        source = RANKING_FILE
        destination = str(uuid.uuid4()) + '.json'
        shutil.copy(source, destination)

operation_list = [
    Operations.add_game,
    Operations.remove_game,
    Operations.create_player,
    Operations.replay_all_games,
    Operations.undo_game,
    Operations.redo_game,
    Operations.get_plot,
    Operations.create_backup
]
try:
    JSON_FILE = json.load(open(RANKING_FILE))
except:
    create_rankings_file()
    JSON_FILE = json.load(open(RANKING_FILE))

if __name__ == '__main__':
    create_rankings_file()
    games = ["PONG", "TITS", "DIE", "BALL", "SNAPPA", "TWENTY ONE 21", "FLIP CUP", "BASEBALL"]
    players = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    for player in players:
        create_player(player)
    for i in range(0, 100):
        arrays = [
            [1, 2, 3, 4],
            [1, 1, 2, 2]
        ]
        choice = np.random.randint(0, 2)
        decision = arrays[choice]
        game = np.random.choice(games, 1)[0]
        player = np.random.choice(players, 4, replace=False)
        resolve_match(game, [
            (player[0], decision.pop(np.random.randint(0, 4)), 6),
            (player[1], decision.pop(np.random.randint(0, 3)), 4),
            (player[2], decision.pop(np.random.randint(0, 2)), 6),
            (player[3], decision.pop(), 4)
        ])
    operation_class = Operations()
    operation_string = ''
    i = 0
    for operation in operation_list:
        operation_string += f'[{i}] {operation.__name__}\n'
        i += 1
    while True:
        i0 = input(f'Choose Operation:\n{operation_string}')
        print('')
        try:
            operation_list[int(i0)](operation_class)
        except NonexistentPlayerException:
            print('Returning to main loop')
            pass
        print_rankings_file()
