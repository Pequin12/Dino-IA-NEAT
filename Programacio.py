"""
Dino Run amb NEAT + Pygame
Fitxer: dino_neat.py
Explicació ràpida:
- Utilitza neat-python (pip install neat-python) i pygame (pip install pygame).
- Executa: python dino_neat.py
- L'script crea un fitxer de configuració NEAT per defecte si no existeix.
- Mostra múltiples agents (quadratets) aprenent a saltar cactus i esquivar ocells.
- A dalt a la dreta hi ha una mini-vista que dibuixa la xarxa neuronal del "millor genoma" de la generació actual.
  Si diverses xarxes controlen agents a la mateixa posició, se'n tria una a l'atzar per mostrar.
- Els obstacles apareixen aleatòriament amb una distribució no constant (intervals aleatoris tipus exponencial) per aproximar el ritme del joc original.

Nota: gràfics simples (quadratets). El codi està pensat per ser llegible i fàcil de modificar.
"""

import pygame
import neat
import os
import random
import math
import time
import pickle

# --------------------------- CONFIG --------------------------------
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 400
FPS = 60
GROUND_Y = 300

# Paràmetres del Dino
DINO_X = 50
DINO_SIZE = 30
JUMP_V = -10
GRAVITY = 0.6

# Obstacles
BASE_SPEED = 8
MIN_SPAWN_MS = 1500
MAX_SPAWN_MS = 2500

# Visualització
POP_MAX_DRAW = 50  # màxim d'agents a dibuixar cada frame (per mantenir rendiment)

# ---------------------------------------------------------------------

# Si el fitxer de configuració de NEAT no existeix, escriure un per defecte.
DEFAULT_NEAT_CONFIG = """

[NEAT]
fitness_criterion     = max
fitness_threshold     = 20000
pop_size              = 3000
reset_on_extinction   = False

[DefaultGenome]
# node activation options
activation_default      = tanh
activation_mutate_rate  = 0.1
activation_options      = tanh sigmoid relu

# node aggregation options
aggregation_default     = sum
aggregation_mutate_rate = 0.1
aggregation_options     = sum

# node bias options
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30.0
bias_min_value          = -30.0
bias_mutate_power       = 0.5
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

# genome compatibility options
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# connection add/remove rates
conn_add_prob           = 0.5
conn_delete_prob        = 0.5

# connection enable options
enabled_default         = True
enabled_mutate_rate     = 0.01

feed_forward            = True
initial_connection      = unconnected

# node add/remove rates
node_add_prob           = 0.2
node_delete_prob        = 0.2

# network parameters
num_hidden              = 0
num_inputs              = 5
num_outputs             = 1

# node response options
response_init_mean      = 1.0
response_init_stdev     = 0.0
response_max_value      = 30.0
response_min_value      = -30.0
response_mutate_power   = 0.0
response_mutate_rate    = 0.0
response_replace_rate   = 0.0

# connection weight options
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_power     = 0.5
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20
species_elitism      = 2

[DefaultReproduction]
elitism            = 2
survival_threshold = 0.2
"""

class Dino:
    def __init__(self):
        self.x = DINO_X
        self.y = GROUND_Y - DINO_SIZE
        self.size = DINO_SIZE
        self.vel_y = 0
        self.is_jumping = False
        self.alive = True
        self.score = 0

    def update(self):
        # física
        self.vel_y += GRAVITY
        self.y += self.vel_y
        if self.y >= GROUND_Y - self.size:
            self.y = GROUND_Y - self.size
            self.vel_y = 0
            self.is_jumping = False

    def jump(self):
        if not self.is_jumping and self.y >= GROUND_Y - self.size - 1:
            self.vel_y = JUMP_V
            self.is_jumping = True

    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.size, self.size)


class Obstacle:
    def __init__(self, kind='cactus'):
        self.kind = kind
        self.speed = BASE_SPEED
        if kind == 'cactus':
            self.width = random.randint(20, 28)
            self.height = random.randint(30, 40)
            self.y = GROUND_Y - self.height
        else:  # ocell
            self.width = 34
            self.height = 24
            # l'ocell vola a diferents altures
            self.y = GROUND_Y - self.height - random.choice([70, 100, 40])
        self.x = WINDOW_WIDTH + random.randint(0, 200)

    def update(self):
        self.x -= self.speed

    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), self.width, self.height)


def spawn_obstacle():
    # imitar aproximadament Dino run mesclant cactus i ocells
    kind = 'cactus' if random.random() < 0.75 else 'bird'
    return Obstacle(kind)


# Funció auxiliar per calcular el temps fins al proper obstacle (aleatoritzat, no constant)
def next_spawn_time_ms():
    # Utilitza distribució exponencial per produir intervals irregulars similars al joc humà
    # mitjana al voltant de 1400ms però limitada
    lam = 1 / 1.4
    val = random.expovariate(lam) * 1000
    return max(MIN_SPAWN_MS, min(MAX_SPAWN_MS, int(val)))


# Dibuixar la visualització petita de la xarxa a dalt a la dreta
def draw_network(surface, genome, config, topleft=(580, 10), size=(210, 140)):
    # Dibuixarem nodes en capes: entrades -> ocults (si n'hi ha) -> sortides
    # Per simplicitat, situar nodes d'entrada verticalment a l'esquerra, sortides a la dreta.
    font = pygame.font.SysFont(None, 12)
    x0, y0 = topleft
    w, h = size

    # recollir nodes
    node_keys = list(genome.nodes.keys())
    inputs = [k for k in node_keys if k < config.genome_config.input_keys[-1] + 1]
    # neat utilitza -1, -2 etc? Per simplificar, utilitzem config
    in_keys = list(config.genome_config.input_keys)
    out_keys = list(config.genome_config.output_keys)

    # obtenir nodes ocults
    hidden_keys = [k for k in node_keys if (k not in in_keys) and (k not in out_keys)]

    layers = [in_keys, hidden_keys, out_keys]
    layer_x = [x0 + 10, x0 + w // 2, x0 + w - 20]
    # calcular espai vertical
    max_nodes_in_layer = max(len(l) for l in layers if len(l) > 0)
    node_positions = {}
    for li, layer in enumerate(layers):
        cnt = len(layer)
        if cnt == 0:
            continue
        spacing = h / (cnt + 1)
        for ii, nk in enumerate(layer):
            posx = layer_x[li]
            posy = int(y0 + (ii + 1) * spacing)
            node_positions[nk] = (posx, posy)
            pygame.draw.circle(surface, (240, 240, 240), (posx, posy), 6)
            # id petit del node
            txt = font.render(str(nk), True, (10, 10, 10))
            surface.blit(txt, (posx - 6, posy - 6))

    # dibuixar connexions
    for cg_key, cg in genome.connections.items():
        if not cg.enabled:
            continue
        in_k = cg.key[0]
        out_k = cg.key[1]
        if in_k in node_positions and out_k in node_positions:
            x1, y1 = node_positions[in_k]
            x2, y2 = node_positions[out_k]
            # el pes mapeja a gruix de línia
            wth = max(1, int(min(4, abs(cg.weight) * 2)))
            col = (0, 200, 0) if cg.weight > 0 else (200, 0, 0)
            pygame.draw.line(surface, col, (x1, y1), (x2, y2), wth)

    # contorn
    pygame.draw.rect(surface, (150, 150, 150), (x0, y0, w, h), 1)
    label = font.render('Millor IA actual', True, (220, 220, 220))
    surface.blit(label, (x0 + 6, y0 + 4))


# Avaluar genomes utilitzant visualització amb pygame
def eval_genomes(genomes, config):
    # configurar pygame
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()

    nets = {}
    dinos = {}
    ge = {}

    # crear objectes per cada genoma
    for gid, genome in genomes:
        genome.fitness = 0.0
        nets[gid] = neat.nn.FeedForwardNetwork.create(genome, config)
        dinos[gid] = Dino()
        ge[gid] = genome

    obstacles = []
    spawn_timer = next_spawn_time_ms()
    elapsed_ms = 0
    generation_time_limit_ms = 60000  # límit dur de 60 segons

    run = True
    while run:
        dt = clock.tick(FPS)
        elapsed_ms += dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                os._exit(0)

        # generar obstacles
        spawn_timer -= dt
        if spawn_timer <= 0:
            obstacles.append(spawn_obstacle())
            spawn_timer = next_spawn_time_ms()

        # actualitzar obstacles
        for ob in obstacles:
            ob.update()
        obstacles = [o for o in obstacles if o.x + o.width > -50]

        # actualitzar dinos
        alive_count = 0
        for gid, dino in list(dinos.items()):
            if not dino.alive:
                continue
            alive_count += 1

            # entrades per la xarxa:
            # 1) distància al proper obstacle
            # 2) amplada de l'obstacle
            # 3) altura de l'obstacle
            # 4) y de l'obstacle (top)
            # 5) velocitat vertical del dino
            # Si no hi ha obstacle, posar valors grans
            if len(obstacles) > 0:
                # triar l'obstacle més proper davant del dino
                next_ob = None
                for ob in obstacles:
                    if ob.x + ob.width >= dino.x:
                        next_ob = ob
                        break
                if next_ob is None:
                    next_ob = obstacles[0]
                dist = max(0.0, next_ob.x - dino.x)
                ob_w = next_ob.width
                ob_h = next_ob.height
                ob_y = next_ob.y
            else:
                dist = 1000.0
                ob_w = 0
                ob_h = 0
                ob_y = 0

            # normalitzar entrades aproximadament
            inp = [dist / WINDOW_WIDTH, ob_w / 100.0, ob_h / 100.0, (GROUND_Y - ob_y) / WINDOW_HEIGHT, (dino.vel_y + 15) / 30.0]

            output = nets[gid].activate(inp)
            # una sortida: probabilitat de salt
            if output[0] > 0.5:
                dino.jump()

            dino.update()

            # col·lisió
            drect = dino.rect()
            dead = False
            for ob in obstacles:
                if drect.colliderect(ob.rect()):
                    dead = True
                    break
            if dead:
                dino.alive = False
                ge[gid].fitness -= 1.0
            else:
                # recompensa per sobreviure i avançar
                ge[gid].fitness += 0.1
                dino.score += 1

        # condició de parada: tots morts o límit de temps
        if alive_count == 0 or elapsed_ms > generation_time_limit_ms:
            run = False

        # Visualització: dibuixar tot
        screen.fill((30, 30, 30))
        # terra
        pygame.draw.line(screen, (200, 200, 200), (0, GROUND_Y), (WINDOW_WIDTH, GROUND_Y), 2)

        # dibuixar obstacles
        for ob in obstacles:
            col = (80, 180, 80) if ob.kind == 'cactus' else (180, 80, 80)
            pygame.draw.rect(screen, col, ob.rect())

        # dibuixar dinos (limitar dibuixats a POP_MAX_DRAW per rendiment)
        drawn = 0
        for gid, dino in dinos.items():
            if not dino.alive:
                continue
            if drawn >= POP_MAX_DRAW:
                break
            pygame.draw.rect(screen, (100, 200, 250), dino.rect())
            drawn += 1

        # --- Seguiment del millor actual ---
        # inicialitzar si no existeix
        if not hasattr(eval_genomes, "current_best_gid"):
            eval_genomes.current_best_gid = None
            eval_genomes.current_best_fitness = -1e9
            eval_genomes.current_best_genome = None

        # comprovar si hi ha un nou líder amb millor fitness
        for gid, genome in genomes:
            if genome.fitness > eval_genomes.current_best_fitness:
                eval_genomes.current_best_fitness = genome.fitness
                eval_genomes.current_best_gid = gid
                eval_genomes.current_best_genome = genome

        # si l'actual líder mor, triar un altre dels vius amb millor fitness
        if eval_genomes.current_best_gid is not None:
            if eval_genomes.current_best_gid in dinos and not dinos[eval_genomes.current_best_gid].alive:
                alive_best = None
                alive_best_fit = -1e9
                for gid, dino in dinos.items():
                    if dino.alive and ge[gid].fitness > alive_best_fit:
                        alive_best_fit = ge[gid].fitness
                        alive_best = gid
                if alive_best is not None:
                    eval_genomes.current_best_gid = alive_best
                    eval_genomes.current_best_fitness = ge[alive_best].fitness
                    eval_genomes.current_best_genome = ge[alive_best]

        # dibuixar la xarxa del millor actual (si existeix)
        if eval_genomes.current_best_genome is not None:
            draw_network(screen, eval_genomes.current_best_genome, config,
                         topleft=(WINDOW_WIDTH - 220, 6), size=(210, 150))
        else:
            pygame.draw.rect(screen, (100, 100, 100), (WINDOW_WIDTH - 220, 6, 210, 150), 1)

        # HUD petit
        font = pygame.font.SysFont(None, 20)
        gen_text = font.render('Gen: {}'.format(eval_genomes.generation), True, (220, 220, 220))
        alive_text = font.render('Vius: {}'.format(alive_count), True, (220, 220, 220))
        screen.blit(gen_text, (10, 10))
        screen.blit(alive_text, (10, 34))

        pygame.display.flip()

    pygame.event.pump()
    # assignar fitness final
    # la llista de genomes ja referencia els genomes, així que el fitness assignat abans persisteix

# guardar número de generació a la funció per poder mostrar-lo
eval_genomes.generation = 0

def run(config_file):
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)

    # reporters (stdout)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # embolicar eval per fer seguiment del número de generació
    def wrapped_eval(genomes, config_in):
        eval_genomes.generation = wrapped_eval.generation
        eval_genomes(genomes, config_in)
        wrapped_eval.generation += 1

    wrapped_eval.generation = 0

    # executar fins a 5000 generacions
    winner = p.run(wrapped_eval, 5000)

    # guardar guanyador
    with open('winner.pkl', 'wb') as f:
        pickle.dump(winner, f)
    print('\nMillor genoma guardat a winner.pkl')


if __name__ == '__main__':
    # assegurar que la configuració existeix
    cfg_path = 'config-feedforward.txt'
    if not os.path.exists(cfg_path):
        with open(cfg_path, 'w') as f:
            f.write(DEFAULT_NEAT_CONFIG)
        print('S’ha escrit la configuració NEAT per defecte a', cfg_path)

    run(cfg_path)
