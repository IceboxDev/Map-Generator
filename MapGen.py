# -*- coding: utf-8 -*-
# TODO Monkey-Patching and method overriding

# Core modules
import scipy.spatial.distance as dist
import matplotlib.pyplot      as pyplot
import numpy                  as np

# In builts
import collections
import matplotlib
import datetime
import operator
import typing
import random
import time
import sys

# Special types
point_type  = typing.Union[tuple, np.ndarray]
container   = typing.Union[tuple, dict]
interval2d  = typing.Tuple[tuple, tuple]


# Class for generation of maps
class Map:

    # Display
    WIDTH           = 10
    HEIGHT          = 5
    WALL_TILE       = "█"
    FLOOR_TILE      = " "
    ENCODING        = "utf-8"
    DEFAULT_DISPLAY = sys.stdout
    COLOR           = pyplot.cm.binary
    IMAGE_FORMATS   = ["png", "pdf", "ps", "eps", "svg"]

    # Mob spawning probabilities
    MOBS_1  = {"Barbarian"      : (3 / 12),
               "Wizard"         : (1 / 12),
               "Paladin"        : (2 / 12),
               "Zombie"         : (3 / 12),
               "Lich"           : (1 / 12),
               "Wraith"         : (2 / 12), }

    # Interactable tiles
    TILES   = {"Impassable"     : {1 : "Wall" ,
                                   4 : "Crate",
                                   5 : "Sign" ,
                                   6 : "Chest", },
               "Passable"       : {0 : "Floor",
                                   2 : "Water",
                                   3 : "Lava" ,
                                   7 : "Plate", }, }

    # Types of walls
    WALLS   = {0 : "stone"    ,
               1 : ""         , }

    # Types of wire
    WIRES   = {0 : "red"      ,
               1 : "green"    ,
               2 : "blue"     ,
               3 : "yellow"   , }

    # Types of map generation
    TYPES   = {1 : "Voronoi Diagram"           ,
               2 : "Cellular Automata"         ,
               3 : "Binary Space Partitioning" ,
               4 : "Random Segment Maze"       ,
               5 : ""                          , }

    # Map initialization
    def __init__(self, height: int, width: int, seed: int = None, levelname: str = ""):

        # Map can only have odd shape (Reason - symmetry for mazes)
        self.height = height + int(not (height % 2))
        self.width  = width  + int(not (width  % 2))

        # Actual map
        self.size   = self.height * self.width
        self.map    = np.zeros([self.height, self.width])

        # Start / End
        self.startx = 0
        self.starty = 0
        self.exitx  = 0
        self.exity  = 0

        # Entity storage
        self.lname = levelname
        self.enemy = {}
        self.signs = {}
        self.doors = {}
        self.wires = {}
        self.ghost = {}

        # Seed can be provided with input to recreate a map
        if not seed:
            self.seed = random.randrange(sys.maxsize)

        else:
            self.seed = seed

        np.random.seed(self.seed)
        random.seed(self.seed)

    # Array entry conversion to display
    @staticmethod
    def _convert(tile_type: int) -> str:

        # 1 Means wall
        if tile_type == 1:
            return Map.WALL_TILE

        # 0 Means floor
        elif tile_type == 0:
            return Map.FLOOR_TILE

    # Click handling while displaying map
    def _display_onclick(self, event: matplotlib.backend_bases.MouseEvent, image: matplotlib.image.AxesImage) -> None:

        # Excepting cases where
        if event.inaxes:

            pos = (npround(event.ydata), npround(event.xdata))

            # Left mouse button
            if   event.button == 1:
                pass

            # Middle mouse button
            elif event.button == 2:
                pass

            # Right moouse button
            elif event.button == 3:
                self.map[pos] = not self.map[pos]

            # Updating map
            image.set_data(self.map)
            pyplot.draw()

    # Sets borders for the map
    def borders(self) -> None:

        self.map[ 0,  :] = 1
        self.map[-1,  :] = 1
        self.map[ :,  0] = 1
        self.map[ :, -1] = 1

    # Returns all neighbouring tiles
    def neighbours(self, y: int, x: int, radius: int = 1, diagonal: bool = False) -> list:

        output = []

        # Eight neighbouring tiles
        if diagonal:

            # Iteration through all tiles
            for c1 in range(max(0, y - radius), min(self.height , y + radius + 1)):
                for c2 in range(max(0, x - radius), min(self.width, x + radius + 1)):
                    if (c1, c2) != (y, x):
                        output.append((c1, c2))

        # Four neighbouring tiles
        else:
            output += [(c1, x) for c1 in range(max(0, y - radius), min(self.height , y + radius + 1))]
            output += [(y, c2) for c2 in range(max(0, x - radius), min(self.width  , x + radius + 1))]
            output.remove((y, x))
            output.remove((y, x))

        return output

    # Fills an area
    def flood_fill(self, start_y: int, start_x: int, fillable: container = TILES["Passable"]) -> list:

        # Points that have been visited already
        visited = []
        targets = [(start_y, start_x)]

        # As long as there's something visitable
        while targets:
            point = targets.pop()

            # That's a new point
            if point not in visited:

                # All neighbouring points that are passable
                for n_point in [i for i in self.neighbours(*point) if self.map[i] in fillable]:
                    targets.append(n_point)

                # The point has been visited
                visited.append(point)

        return visited

    # Breadth first search / Deapth first search
    def search(self, start_y: int, start_x: int, exit_function: typing.Callable, dfs: bool = False) -> point_type:

        # Points that have been visited already
        visited = []
        targets = [(start_y, start_x)]

        # As long as there's something visitable
        while targets:

            # Points are chosen by depth or by breadth
            point = targets.pop(-1 * dfs)

            # That's a new point
            if point not in visited:

                # All neighbouring points
                for n_point in self.neighbours(*point):

                    # Target reached
                    if exit_function(n_point):
                        return n_point

                    # Add new potential targets
                    else:
                        targets.append(n_point)

                # The point has been visited
                visited.append(point)

    # Chooses random points on the map
    def random_points(self, n: int, margin: int = 0, step: int = 1) -> np.ndarray:

        available_height = (self.height - margin * 2) // step + (self.height - margin * 2) % step
        available_width  = (self.width  - margin * 2) // step + (self.width  - margin * 2) % step
        available_points = np.empty([available_height, available_width], dtype=("int32", 2))

        # A point at some index is equal to the index
        for c1 in range(margin, self.height - margin, step):
            for c2 in range(margin, self.width - margin, step):
                available_points[(c1 - margin) // step, (c2 - margin) // step] = (c1, c2)

        # Array conversion
        available_points = np.concatenate(available_points)
        array_indices    = random.sample(range(len(available_points)), n)
        array_points     = np.array([available_points[index] for index in array_indices])

        return array_points

    # Sets multiple points on the map to some value
    def set_points(self, points: np.ndarray, value: int) -> None:

        # Refomating from points to indices
        t = np.transpose(points)
        s = np.split(t, 2)

        self.map[s] = value

    # Displays an iteractive map or stores it to a file
    def display(self, f: str = None) -> None:

        # If filename was provided, store to file
        if f:
            name, extension = f.split(".")

            # Export as an image file
            if extension in Map.IMAGE_FORMATS:
                fig, ax = pyplot.subplots(figsize=(Map.WIDTH, Map.HEIGHT))
                ax.imshow(self.map, cmap=Map.COLOR)
                pyplot.savefig(f, format=extension)

            # Export as a text file
            else:
                with open(f, "w", encoding=Map.ENCODING) as f:
                    for row in self.map:
                        print(*map(Map._convert, row), sep="", file=f)

        # Display interactive map
        else:
            fig, ax = pyplot.subplots(figsize=(Map.WIDTH, Map.HEIGHT))
            image = ax.imshow(self.map, cmap=Map.COLOR, vmin=0, vmax=1)
            fig.canvas.mpl_connect("button_press_event", lambda event: self._display_onclick(event, image))
            ax.format_coord = lambda *coord: "X = %s, Y = %s" % tuple(map(npround, coord))
            pyplot.show()

    # Exports map as txt
    # noinspection PyStringFormat
    def export(self, filename: str, walltype: int = 0) -> None:

        walls   = []
        water   = []
        lava    = []

        for c1 in range(self.height):
            for c2 in range(self.width):
                tile = self.map[c1, c2]

                if   tile == 1:
                    walls.append((c2, c1))

                elif tile == 2:
                    water.append((c2, c1))

                elif tile == 3:
                    lava.append( (c2, c1))

        with open(filename, "w", encoding=Map.ENCODING) as output_file:

            # levelname[string]
            print(self.lname, end="\n\n", file=output_file)

            # levelwidth[int] levelheight[int]
            print(self.width, self.height, end="\n\n", file=output_file)

            # playerstartx[int] playerstarty[int]
            print(self.startx, self.starty, end="\n\n", file=output_file)

            # exitx[int] exity[int]
            print(self.exitx, self.exity, end="\n\n", file=output_file)

            # wallcount[int] (tiek next eiluciu bus dedicatinta walls)
            # wallx[int] wally[int] walltype[string] (kolkas tik "stone") wallactivatable[bool] walltoggleable[bool]
            # walls yra automatiskai generatinamos aplink map,
            # todel cia reikia irasyt tik tas walls, kurios bus paciam level pastatytos
            print(len(walls), file=output_file)

            print(*map(lambda coords: "%s %s %s %s %s"
               % (*coords, Map.WALLS[walltype], *self.ghost.pop(coords[::-1], (0, 0))), walls),
                  sep="\n", end="\n" + "\n" * (len(walls) > 0), file=output_file)

            # watercount[int] (water tiles)
            # waterx[int] watery[int]
            print(len(water), file=output_file)

            print(*map(lambda coords: "%s %s" % coords, water),
                  sep="\n", end="\n" + "\n" * (len(water) > 0), file=output_file)

            # lavacount[int]
            # lavax[int] lavay[int]
            print(len(lava), file=output_file)

            print(*map(lambda coords: "%s %s" % coords, lava),
                  sep="\n", end="\n" + "\n" * (len(lava) > 0), file=output_file)

            # dirt
            # 0 (sitas 2 eilutes tsg visada imesk)
            print("dirt", 0, sep="\n", end="\n\n", file=output_file)

            # wirecount[int]
            # wirex[int] wirey[int] wirecolor[string] isgate[bool] gatetype[string] gatedirection[string]
            # ("up", "down", "left", "right")
            print(0, end="\n\n", file=output_file)

            # cratecount[int]
            # cratex[int] cratey[int]
            print(0, end="\n\n", file=output_file)

            # pressureplatecount[int]
            # ppx[int] ppy[int] pptype[string]("wooden" || "steel" || "gold")
            # wooden activatina mobs/player/gold/crates | steel activatina mobs/player/crates | gold activatina tik gold
            print(0, end="\n\n", file=output_file)

            # lampcount[int]
            # lampx[int] lampy[int] lamptype[string] (kolkas tik "led") lamptoggleable[bool]
            print(0, end="\n\n", file=output_file)

            # doorcount[int]
            # doorx[int] doory[int] doordirection[string]
            print(0, end="\n\n", file=output_file)

            # signcount[int]
            # signx[int] signy[int] signtext[string] (tekste /n reiskia newline nes \n neskaito is .txt failo)
            print(0, end="\n\n", file=output_file)

            # 0 (just that)
            print(0, end="\n\n", file=output_file)

            # (((((( DISCLAIMER ))))))
            # visu mob type yra tik "small"

            # barbariancount[int]
            # bbx[int] bby[int] bbtype[string]
            enemy1 = [p for p in self.enemy if self.enemy[p] == "Barbarian"]
            print(len(enemy1), *map(lambda p: "%s %s small" % p[::-1], enemy1),
                  sep="\n", end="\n" + "\n" * (len(enemy1) > 0), file=output_file)

            # wizardcount[int]
            # wx[int] wy[int] wtype[string]
            enemy2 = [p for p in self.enemy if self.enemy[p] == "Wizard"]
            print(len(enemy2), *map(lambda p: "%s %s small" % p[::-1], enemy2),
                  sep="\n", end="\n" + "\n" * (len(enemy2) > 0), file=output_file)

            # paladincount[int]
            # px[int] py[int] ptype[string]
            enemy3 = [p for p in self.enemy if self.enemy[p] == "Paladin"]
            print(len(enemy3), *map(lambda p: "%s %s small" % p[::-1], enemy3),
                  sep="\n", end="\n" + "\n" * (len(enemy3) > 0), file=output_file)

            # zombiecount[int]
            # zx[int] zy[int] ztype[string]
            enemy4 = [p for p in self.enemy if self.enemy[p] == "Zombie"]
            print(len(enemy4), *map(lambda p: "%s %s small" % p[::-1], enemy4),
                  sep="\n", end="\n" + "\n" * (len(enemy4) > 0), file=output_file)

            # lichcount[int]
            # lx[int] ly[int] ltype[string]
            enemy5 = [p for p in self.enemy if self.enemy[p] == "Lich"]
            print(len(enemy5), *map(lambda p: "%s %s small" % p[::-1], enemy5),
                  sep="\n", end="\n" + "\n" * (len(enemy5) > 0), file=output_file)

            # wraithcount[int]
            # wx[int] wy[int] wtype[string]
            enemy6 = [p for p in self.enemy if self.enemy[p] == "Wraith"]
            print(len(enemy6), *map(lambda p: "%s %s small" % p[::-1], enemy6),
                  sep="\n", end="\n" + "\n" * (len(enemy6) > 0), file=output_file)

            # Finalization
            print("// The following map was proceduraly generated and exported",
                  "// By map generation software made by Icebox: https://github.com/akys200",
                  sep="\n", end="\n//\n" , file=output_file)

            print("// MAP SEED       :", self.seed, file=output_file)
            print("// GENERATED      :", str(datetime.datetime.now()).split('.')[0], file=output_file)


# Class for generation of maps using the voronoi diagram
class Voronoi(Map):

    # Voronoi
    # A Voronoi diagram is a partitioning of a plane into regions
    # based on distance to points in a specific subset of the plane.
    # Every reagion will be a separated room of the map
    # The V_ROOM_SIZE will be the average (manhattan)
    # area of each region including it's walls

    ROOM_SIZE = 500

    # Initialization function
    def __init__(self, height: int, width: int, seed: int = None, levelname: str = ""):
        super().__init__(height, width, seed=seed, levelname=levelname)

        # Key variables
        room_count  = max(1, round(self.size / Voronoi.ROOM_SIZE))
        rooms       = self.random_points(room_count, margin=1)
        areas       = collections.defaultdict(list)

        # Generating vonoroi
        for c1 in range(self.height):
            for c2 in range(self.width):
                areas[tuple(min(rooms, key=lambda room: dist.cityblock(room, (c1, c2))))].append((c1, c2))

        # Placing walls
        for area in areas:
            for row in set(point[1] for point in areas[area]):
                points  = [point for point in areas[area] if point[1] == row]
                minimum = min(points, key=lambda p: p[0])
                maximum = max(points, key=lambda p: p[0])

                self.map[minimum] = 1
                self.map[maximum] = 1

            for column in set(point[0] for point in areas[area]):
                points  = [point for point in areas[area] if point[0] == column]
                minimum = min(points, key=lambda p: p[1])
                maximum = max(points, key=lambda p: p[1])

                self.map[minimum] = 1
                self.map[maximum] = 1

        # Removing borders
        for area in [area for area in areas]:
            for point in areas[area][:]:
                if self.map[point]:
                    areas[area].remove(point)

            if not areas[area]:
                areas.pop(area)

        # Parameters
        self.areas = areas


# Class for generation of maps using cellular automata
class CellularAutomata(Map):

    # Cellular Automata
    # Alive cells represent walls while dead cells represent floor
    # The higher the cell birth parameter, the less likely it is for
    # a wall to survive on the map (higher value will result in more
    # free explorable area)
    # The higher the cell death parameter, the less likely it is for
    # a floor tile to become a wall (lower value will result in more
    # colons and smoother cave endings)
    # The higher the initialisation probability, the bigger wall to
    # floor ratio on initial map creation. (higher value will result
    # in more scattered walls and columns in the end result)
    # The higher the depth, the more simulations will be performed to
    # achieve perfect state that is equal to birth/death stats
    # (lower value will result in more natural caves)
    # Cave smoothness is the birth-death delta (lower delta value
    # will result in smoother cave endings)
    # birth > death delta will give smoothness
    # death > birth delta will give sharpness

    CELL_BIRTH  = 4
    CELL_DEATH  = 4
    INIT_CELL   = 0.45
    SIMULATIONS = 5

    # Initialization function
    def __init__(self, height: int, width: int, seed: int = None, levelname: str = ""):
        super().__init__(height, width, seed=seed, levelname=levelname)

        # Information
        self.main   = []
        self.roomed = False
        self.holed  = False
        self.colmd  = 0

        # Initial points
        point_count = round(self.size * CellularAutomata.INIT_CELL)
        self.set_points(self.random_points(point_count), 1)
        self.borders()

        # Simulating
        for _ in range(CellularAutomata.SIMULATIONS):

            new_map = np.empty([self.height, self.width])

            for c1 in range(1, self.height - 1):
                for c2 in range(1, self.width - 1):
                    n = self.neighbouring_walls((c1, c2), 1)

                    if self.map[c1][c2]:
                        if n < CellularAutomata.CELL_DEATH:
                            new_map[c1][c2] = 0

                        else:
                            new_map[c1][c2] = 1

                    else:
                        if n > CellularAutomata.CELL_BIRTH:
                            new_map[c1][c2] = 1

                        else:
                            new_map[c1][c2] = 0

            self.map = np.copy(new_map)
            self.borders()

    # Checks how many neighbouring tiles are equal to target
    def neighbouring_walls(self, point: point_type, target: int) -> int:

        output = 0

        # Iterating through all neighbouring tiles
        for c1 in range(max(0, point[0] - 1), min(self.height, point[0] + 2)):
            for c2 in range(max(0, point[1] - 1), min(self.width, point[1] + 2)):
                if (c1, c2) != point:
                    output += self.map[c1, c2] == target

        return output

    # Removes enclosed areas and columns that are too small
    def roomify(self, remove_holes: bool = True, remove_columns: int = 5) -> None:

        # Parameters
        self.roomed = True
        self.holed  = remove_holes
        self.colmd  = remove_columns

        # Arrays where all the areas will be stored
        holes   = []
        columns = []

        # Iterating through all points
        for c1 in range(1, self.height-1):
            for c2 in range(1, self.width-1):

                # Hole handling
                if self.map[c1, c2] == 0:
                    if  all((c1, c2) not in room for room in holes):
                        holes.append(self.flood_fill(c1, c2, (0,)))

                # Column handling
                else:
                    if all((c1, c2) not in room for room in columns):
                        columns.append(self.flood_fill(c1, c2, (1,)))

        # Main cave info
        self.main = max(holes, key=lambda h: len(h))
        main_size = len(self.main)

        # Removing holes
        if remove_holes:
            for hole in holes:
                if len(hole) < main_size:
                    self.set_points(np.array(hole), 1)

        # Removing columns
        for column in columns:
            if len(column) <= remove_columns:
                self.set_points(np.array(column), 0)

        # Finding a good starting point
        start_point = self.search(0, 0, lambda p: self.map[p] == 0)
        exit_point  = self.search(self.height - 1, self.width - 1, lambda p: self.map[p] == 0)

        # Settinig the points
        self.startx = start_point[1]
        self.starty = start_point[0]
        self.exitx  = exit_point[1]
        self.exity  = exit_point[0]

        # Area around spawnpoint will have no mobs
        for safe_point in self.neighbours(*start_point, radius=5, diagonal=True):

            # Try removing the point from the main cave
            try:
                self.main.remove(safe_point)

            # Point is a wall
            except ValueError:
                pass

    # Adds mobs to the cave
    def mobify(self, probability: float = 0.10, mob_group: dict = Map.MOBS_1) -> None:

        # All mobs and their respective probabilities
        mob_t = []
        mob_p = []

        for key, value in mob_group.items():
            mob_t.append(key)
            mob_p.append(value)

        # Places where mobs will spawn
        mob_c = round(len(self.main) * probability)
        mob_l = random.sample(self.main, mob_c)
        types = np.random.choice(mob_t, size=mob_c, p=mob_p)

        # Setting their positions
        for index in range(mob_c):
            self.enemy[mob_l[index]] = types[index]

    # Exports map as txt
    def export(self, filename: str, walltype: int = 0):
        super().export(filename, walltype)

        with open(filename, "a", encoding=Map.ENCODING) as output_file:

            # Class paramters
            print("// CELL_BIRTH     :", CellularAutomata.CELL_BIRTH  , file=output_file)
            print("// CELL_DEATH     :", CellularAutomata.CELL_DEATH  , file=output_file)
            print("// INIT_CELL      :", CellularAutomata.INIT_CELL   , file=output_file)
            print("// SIMULATIONS    :", CellularAutomata.SIMULATIONS , file=output_file)

            # Method parameters
            print("// ROOMIFY CALLED :", self.roomed, file=output_file)
            print("// HOLE REMOVAL   :", self.holed , file=output_file)
            print("// COLUMN REMOVAL :", self.colmd , file=output_file)


# Class for generation of maps using binary space partitioning
class BSP(Map):

    # Binary Space Partitioning
    # Binary space partitioning is a method for recursively subdividing
    # a space into convex sets by hyperplanes.
    # The starting map will be divided into smaller pieces recursively
    # Piece will NOT be divided anymore if it's size is below B_ROOM_SIZE

    ROOM_SIZE = 1000

    # Initialization function
    def __init__(self, height, width, seed=None, levelname: str = ""):
        super().__init__(height, width, seed=seed, levelname=levelname)

        # Core
        self.map    = np.ones([self.height, self.width])
        self.tree   = BSP._partition((1, self.height-2), (1, self.width-2), BSP.ROOM_SIZE)

        # Map carving
        for leaf in BSP.tree_iterate(self.tree):
            for c1 in       range(leaf[0][0], leaf[0][1]+1):
                for c2 in   range(leaf[1][0], leaf[1][1]+1):
                    self.map[c1][c2] = 0

    # Recursively subdivides a plane
    @staticmethod
    def _partition(height_interval: tuple, width_interval: tuple, area_limit: int) -> container:

        # Information
        len_h = abs(operator.sub(*height_interval)) + 1
        len_w = abs(operator.sub(*width_interval)) + 1
        info = (height_interval, width_interval)
        area = len_h * len_w
        cut = random.uniform(0.25, 0.75)

        # No more cuts
        if area < area_limit:
            return info

        # Vertical cut
        if ((2 * len_h < len_w) or random.random() < 0.5) and not (2 * len_w < len_h):
            wall = width_interval[0] + round(len_w * cut)

            return {info: (BSP._partition(height_interval, (width_interval[0], wall - 1), area_limit),
                           BSP._partition(height_interval, (wall + 1, width_interval[1]), area_limit))}

        # Horizontal cut
        else:
            wall = height_interval[0] + round(len_h * cut)

            return {info: (BSP._partition((height_interval[0], wall - 1), width_interval, area_limit),
                           BSP._partition((wall + 1, height_interval[1]), width_interval, area_limit))}

    # Iterate through a tree in a form of dictionary
    @staticmethod
    def tree_iterate(dictionary:  dict) -> interval2d:
        for key in dictionary:

            if type(dictionary[key][0]) == dict:
                yield from BSP.tree_iterate(dictionary[key][0])

            else:
                yield dictionary[key][0]

            if type(dictionary[key][1]) == dict:
                yield from BSP.tree_iterate(dictionary[key][1])

            else:
                yield dictionary[key][1]


# Class for generation of maps using random maze segmentation
class RSM(Map):
    
    # Random Segment Maze
    # DENSITY will determine how many random points will be scattered
    # across the map upon initial map creation
    # COMPLEXITY will affect the mutation size of each random point
    # Higher density and lower complexity will mean less open
    # rooms and free spaces and more intersections
    # Higher complexity and lower density will mean less open
    # rooms and longers paths between intersections
    # Lower density and complexity will mean more open rooms
    # COMPLEXITY * 2 + 1 is the maximum length of each segment

    COMPLEXITY  = 15
    DENSITY     = 0.25

    # Initialization function
    def __init__(self, height: int, width: int, seed: int = None, levelname: str = ""):
        super().__init__(height, width, seed=seed, levelname=levelname)

        # Initial changes
        point_count = round((self.width + 1) * (self.height + 1) // 4 * RSM.DENSITY)
        random_pts  = self.random_points(point_count, step=2)
        self.borders()

        # Room information
        self.rooms  = []
        self.ones   = []

        # Converting each random point to a segment
        for random_point in random_pts:

            y, x            = random_point
            self.map[y, x]  = 1

            # Lenghtening each point by 2 for each point of complexity
            for _ in range(RSM.COMPLEXITY):
                nbs = []

                # Finding valid neighbours
                if x > 1:
                    nbs.append((y, x - 2))
                if x < self.width  - 2:
                    nbs.append((y, x + 2))
                if y > 1:
                    nbs.append((y - 2, x))
                if y < self.height - 2:
                    nbs.append((y + 2, x))

                # Expansion
                while nbs:
                    random.shuffle(nbs)
                    ny, nx = nbs.pop()

                    if  self.map[ny, nx] == 0:
                        self.map[ny, nx] = 1
                        self.map[ny + (y - ny) // 2, nx + (x - nx) // 2] = 1
                        x, y = nx, ny
                        break

                # Expoansion impossible, exit early
                else:
                    break

        points  = []

        # All points that could have a wall, but don't
        for c1 in range(0, self.height, 2):
            for c2 in range(0, self.width, 2):
                if self.map[c1, c2] == 0:
                    points.append((c2, c1))

        # Those points become rooms
        while points:
            current = [points.pop()]
            room    = current[:]

            while current:
                current_point = current.pop()
                for c1 in range(current_point[0]-2, current_point[0]+3, 2):
                    for c2 in range(current_point[1]-2, current_point[1]+3, 2):
                        if (c1, c2) in points:
                            points.remove(  (c1, c2))
                            current.append( (c1, c2))
                            room.append(    (c1, c2))

            self.rooms.append(room)

        self.startx = 1
        self.starty = 1
        self.exitx  = self.width  - 2
        self.exity  = self.height - 2

    # Creates rooms in the maze (0 for doors, 1 for activatable walls)
    def roomify(self) -> None:

        rooms = []
        ones  = []

        for room in self.rooms:

            # Room of size 3x3
            if len(room) == 1:
                ones.append(((room[0][0]-1, room[0][1]-1), (room[0][0]+1, room[0][1]+1)))

            # Bigger room
            else:
                max_x = max(room, key=lambda pt: pt[0])[0] + 1
                min_x = min(room, key=lambda pt: pt[0])[0] - 1
                max_y = max(room, key=lambda pt: pt[1])[1] + 1
                min_y = min(room, key=lambda pt: pt[1])[1] - 1

                for x in range(min_x, max_x + 1):
                    for y in range(min_y, max_y + 1):
                        self.map[y, x] = 0

                rooms.append(((min_x, min_y), (max_x, max_y)))

        # Storing new room information
        self.ones = ones
        self.rooms = rooms


# Special form of rounding for numpy float types
def npround(integer: np.float) -> int:
    return int(round(integer))


print("Starting")

a = CellularAutomata(50, 100, levelname="test")
a.roomify()
a.mobify()
a.display()
a.export("test.txt", 0)

print("Done")
