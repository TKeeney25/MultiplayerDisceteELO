from copiedMultielo.multielo import MultiElo
import json
import numpy as np
import matplotlib.pyplot as plt
import difflib

ELO = MultiElo()
RANKING_FILE = 'rankings.json'



class NonexistentPlayerException(Exception):
    pass


def add_to_commit_log(players, places, old_elo, new_elo):
    JSON_FILE['COMMIT_LOG'].append({
        'PLAYERS': players,
        'PLACES': places,
        'DELTA_ELO': list(new_elo - old_elo)
    })
    dump_json_data()


def add_to_redo_log(players, places):
    JSON_FILE['REDO_LOG'].append({
        'PLAYERS': players,
        'PLACES': places,
    })
    dump_json_data()


def resolve_team_match(team_placement):
    all_team_members = []
    all_team_elos = []
    for team in team_placement:
        team_elo = 0
        team_members = []
        for player in team:
            team_elo += player[1]
            team_members.append(player[0])
        all_team_members.append(team_members)
        all_team_elos.append(team_elo)
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
        norm = np.linalg.norm(np_mem_gain) * (np_mem_gain[0] / np.abs(np_mem_gain[0]))
        np_mem_gain = (np_mem_gain/norm) * ratings_diff[i_team] + np.array(mem_old)
        new_adjusted_ratings = np.concatenate((new_adjusted_ratings, np_mem_gain))
    return new_adjusted_ratings


def resolve_match(player_tuples):
    elo_array = np.array([])
    player_tuples.sort(key=lambda x: x[1])
    players = []
    places = []
    team_placement = list(np.empty(player_tuples[-1][1]))
    for item in range(0, len(team_placement)):
        team_placement[item] = []
    for arg in player_tuples:
        players_rating = get_players_rating(arg[0])
        elo_array = np.append(elo_array, [players_rating[1]])
        players.append(players_rating[0])
        places.append(arg[1])
        team_placement[arg[1] - 1].append(players_rating)
    if len(team_placement) < len(players):
        new_ratings = resolve_team_match(team_placement)
    else:
        new_ratings = ELO.get_new_ratings(elo_array)
    for i in range(0, len(players)):
        set_players_rating(players[i], new_ratings[i])
    add_to_commit_log(players, places, elo_array, new_ratings)


def get_players_rating(name: str) -> (str, float):
    name = name.upper()
    try:
        return name, JSON_FILE['PLAYERS'][name]['ELO']
    except:
        names = list(JSON_FILE['PLAYERS'].keys())
        if len(names) != 0:
            closest_match = difflib.get_close_matches(name, names, n=1, cutoff=0)[0]
            response = input(f'You inputted {name}, did you mean {closest_match}? (Y/N) ').upper()
        else:
            response = None
        if response == 'Y':
            return closest_match, JSON_FILE['PLAYERS'][closest_match]['ELO']
        else:
            response = input(f'Options:\n0: Create New Player Called {name}\n1: Enter Different Name\n2: Return\n')
            if response == '0':
                create_player(name)
                return name, JSON_FILE['PLAYERS'][name]['ELO']
            elif response == '1':
                new_name = input('Please input the different name: ')
                return get_players_rating(new_name)
            else:
                raise NonexistentPlayerException(f'Player {name} is not found in the JSON file.')


def set_players_rating(name: str, elo: float):
    name = name.upper()
    JSON_FILE['PLAYERS'][name]['ELO'] = elo
    dump_json_data()


def create_player(name: str):
    name = name.upper()
    default_player = {
        'ELO': 1000.0
    }
    JSON_FILE['PLAYERS'][name] = default_player
    dump_json_data()


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


def get_plot():
    player_and_score = []
    for player in JSON_FILE['PLAYERS']:
        player_and_score.append((player, get_players_rating(player)[1]))
    player_and_score.sort(key=lambda x: x[1], reverse=True)

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
        i1 = input('Number of Players: ')
        players = []
        for i in range(0, int(i1)):
            player = input('Player Name: ')
            place = int(input('Place: '))
            print('')
            players.append((player, place))
        resolve_match(players)

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
                player_delta = last_game['DELTA_ELO'][i]
                set_players_rating(player_name, get_players_rating(player_name)[1] - player_delta)
            if add_to_redo:
                add_to_redo_log(last_game['PLAYERS'], last_game['PLACES'])
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
            for i in range(0, len(last_game['PLAYERS'])):
                player_name = last_game['PLAYERS'][i]
                player_place = last_game['PLACES'][i]
                player_tuples.append((player_name, player_place))
            resolve_match(player_tuples)
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

    def get_plot(self):
        get_plot()

operation_list = [
    Operations.add_game,
    Operations.remove_game,
    Operations.create_player,
    Operations.replay_all_games,
    Operations.undo_game,
    Operations.redo_game,
    Operations.get_plot
]
create_rankings_file()
try:
    JSON_FILE = json.load(open(RANKING_FILE))
except:
    create_rankings_file()
    JSON_FILE = json.load(open(RANKING_FILE))

if __name__ == '__main__':
    create_rankings_file()
    create_player('A')
    create_player('B')
    create_player('C')
    create_player('D')
    for i in range(0, 10):
        arrays = [
            [1,2,3,4],
            [1,1,2,2]
        ]
        choice = np.random.randint(0, 2)
        decision = arrays[choice]
        resolve_match([
            ('A', decision.pop(np.random.randint(0,4))),
            ('B', decision.pop(np.random.randint(0,3))),
            ('C', decision.pop(np.random.randint(0,2))),
            ('D', decision.pop())
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
        get_plot()
        print_rankings_file()