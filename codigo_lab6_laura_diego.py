# Laboratorio realizado por: Laura Valles Ramírez y Diego Barba Arévalo

import pygame
import numpy as np
import random
import matplotlib.pyplot as plt 

# =============================================================================
# 1. CONFIGURACIÓN Y PARÁMETROS DEL ENTORNO
# =============================================================================
# Dimensiones del grid y de la ventana gráfica
ANCHO_CELDA = 80
COLS = 8 
FILAS = 6
MARGEN_TABLERO = 70
ANCHO_VENTANA = COLS * ANCHO_CELDA + 2 * MARGEN_TABLERO
ALTO_INFO_PANEL = 100
ALTO_VENTANA = FILAS * ANCHO_CELDA + ALTO_INFO_PANEL + 2 * MARGEN_TABLERO 
FPS = 500

# Definición de colores para pintar el tablero y los elementos
BLANCO = (255, 255, 255)
NEGRO = (0, 0, 0)
ROJO = (255, 100, 100)
VERDE = (95, 115, 67)
AMARILLO = (255, 255, 0)
MARRON_SUELO = (217, 187, 169) 
MARRON_OBSTACULO = (47, 22, 2) 

# =============================================================================
# 2. DEFINICIÓN DEL MAPA
# =============================================================================
# Matriz que representa la cocina.
# P = Pasillo, O = Obstáculo
# Códigos de 2 letras = Estaciones de trabajo
MAPA = [
    ["PA", "CA", "QU", "TO", "LE", "BA", "O", "TR"],
    ["P", "P", "P", "P", "P", "P", "P", "P"],
    ["O", "P", "P", "O", "HO", "O", "P", "P"],
    ["CO", "P", "P", "O", "O", "O", "P", "P"],
    ["O", "P", "P", "P", "P", "P", "P", "P"],
    ["P", "P", "O", "PL", "O", "P", "P", "EN"]
]

# Diccionario para traducir los códigos a iconos visuales
EMOJIS = {
    "P": "", "O": "", 
    "PA": "🍞", "CA": "🥩", "QU": "🧀", "TO": "🍅", "LE": "🥬", "BA": "🥓",
    "CO": "🔪", "HO": "♨️", "EN": "📥", "PL": "🍽️", "TR": "🗑️",
    "POLLITO": "🐥", "EXITO": "🐣", "BASURA": "🤢", "plato_lleno": "🍔"
}

# Nombres legibles para la interfaz de usuario (Receta)
NOMBRES_COMPLETOS = {
    "pan": "Pan", "carne": "Carne", "queso": "Queso", 
    "tomate": "Tomate", "lechuga": "Lechuga", "bacon": "Bacon",
    "recoger_plato": "Recoger", "entrega": "Entregar"
}

# Pre-cálculo de coordenadas: Guardamos dónde está cada cosa para no buscarlo en cada frame
COORDENADAS_ESTACIONES = {}
for r in range(FILAS):
    for c in range(COLS):
        item = MAPA[r][c]
        if item != "P" and item != "O":
            COORDENADAS_ESTACIONES[item] = (r, c)

# REGLAS DEL DOMINIO:
# Aquí definimos la lógica de cada ingrediente: dónde se coge, qué se le hace y en qué se convierte.
INFO_INGREDIENTES = {
    "pan":   {"loc": "PA", "proceso": "cortar", "final": "pan_cortado"},
    "carne": {"loc": "CA", "proceso": "cocinar", "final": "carne_cocinado"},
    "queso": {"loc": "QU", "proceso": None, "final": "queso"},
    "tomate":{"loc": "TO", "proceso": "cortar", "final": "tomate_cortado"},
    "lechuga":{"loc": "LE", "proceso": "cortar", "final": "lechuga_cortado"},
    "bacon": {"loc": "BA", "proceso": "cocinar", "final": "bacon_cocinado"},
    
    "recoger_plato": {"loc": "PL", "proceso": None, "final": "plato_lleno"},
    "entrega":       {"loc": "EN", "proceso": None, "final": "fin"}
}

# ESPACIO DE ACCIONES: Las 6 cosas que puede decidir el agente
ACCIONES = ["ARRIBA", "ABAJO", "IZQ", "DER", "INTERACTUAR", "ESPERAR"]

# =============================================================================
# 3. CLASE DEL ENTORNO
# =============================================================================
class Environment:
    """
    Clase que representa el entorno (Environment) del Problema de Decisión de Markov (MDP).
    Gestiona el estado, las transiciones y las recompensas.
    """
    def __init__(self):
        self.action_space_n = len(ACCIONES)
        self.meta_hamburguesas = 1 # Empezamos con el objetivo fácil (Curriculum Learning)
        self.reset()

    def generar_receta(self, nivel_sub_episodio):
        """
        Sistema de Curriculum Learning:
        Genera recetas de complejidad creciente.
        Nivel 0: Pan + Carne.
        Niveles > 0: Se añaden ingredientes extra progresivamente.
        """
        receta = ["pan", "carne"]
        orden_extras = ["queso", "bacon", "tomate", "lechuga"]
        
        if nivel_sub_episodio > 0:
            receta.extend(orden_extras[:nivel_sub_episodio])
        
        receta.extend(["recoger_plato", "entrega"])
        return receta

    def reset(self):
        """Reinicia el entorno para empezar un nuevo episodio limpio."""
        # TIME LIMIT: Límite de pasos para evitar bucles infinitos.
        self.max_steps = self.meta_hamburguesas * 400 
        self.nivel_actual_episodio = 0 
        self.receta = self.generar_receta(self.nivel_actual_episodio)
        self.paso_actual = 0
        self.agent_pos = [1, 1] 
        self.mano = None 
        self.sarten = {"item": None, "tiempo": 0} 
        self.plato_contenidos = [] 
        self.steps_taken = 0
        self.done = False
        return self.get_state()

    def get_state(self):
        """
        DEFINICIÓN DEL ESPACIO DE ESTADOS
        Para Q-Learning, el agente necesita saber todo lo relevante para tomar una decisión.
        Discretizamos la información clave para la Q-Table.
        """
        sarten_ocupada = 1 if self.sarten["item"] is not None else 0
        sarten_lista = 1 if self.sarten["tiempo"] >= 2 else 0
        sarten_quemada = 1 if self.sarten["tiempo"] > 5 else 0
        es_error = 1 if self._item_en_mano_es_inutil() else 0
        
        return (
            tuple(self.agent_pos),      # DATO 1: ¿DÓNDE ESTOY? (Coordenadas X,Y)
            self.paso_actual,           # DATO 2: ¿QUÉ TENGO QUE HACER? (Paso de la receta)
            self.mano,                  # DATO 3: ¿QUÉ LLEVO EN LA MANO?
            sarten_ocupada,             # DATO 4: Estado Sartén
            sarten_lista,               # DATO 5: Estado Sartén
            sarten_quemada,             # DATO 6: Estado Sartén
            es_error,                   # DATO 7: ¿La he liado?
            self.nivel_actual_episodio  # DATO 8: Dificultad
        )

    def _item_en_mano_es_inutil(self):
        """Función auxiliar lógica para ayudar al agente a saber si lleva basura o algo incorrecto."""
        if self.mano is None: return False
        if self.mano == "BASURA": return True
        if self.mano == "plato_lleno": return False 

        if self.paso_actual >= len(self.receta): return True
        
        objetivo_actual = self.receta[self.paso_actual]
        
        if objetivo_actual in ["recoger_plato", "entrega"]:
            return True if self.mano != "plato_lleno" else False

        info = INFO_INGREDIENTES[objetivo_actual]
        item_final = info["final"]
        
        if self.mano == item_final: return False
        
        if self.mano == objetivo_actual:
            if info["proceso"] == "cocinar" and self.sarten["tiempo"] > 5:
                return True 
            return False
            
        return True

    def get_target_pos(self):
        """
        HEURÍSTICA PARA REWARD SHAPING
        Calcula la coordenada óptima a la que debería ir el agente.
        Se usa para dar recompensas parciales por acercarse al objetivo.
        """
        if self._item_en_mano_es_inutil():
            return COORDENADAS_ESTACIONES["TR"] # Ir a la basura

        if self.sarten["tiempo"] > 5 and self.mano is None:
             return COORDENADAS_ESTACIONES["HO"] # Ir a limpiar sartén

        if self.paso_actual >= len(self.receta): return (0,0)
        
        objetivo_key = self.receta[self.paso_actual]
        info = INFO_INGREDIENTES[objetivo_key]

        if objetivo_key == "entrega": return COORDENADAS_ESTACIONES["EN"]
        if objetivo_key == "recoger_plato": return COORDENADAS_ESTACIONES["PL"]

        item_final_necesario = info["final"]

        # Lógica de navegación a estaciones intermedias (cortar/cocinar/plato)
        if self.mano == item_final_necesario: return COORDENADAS_ESTACIONES["PL"]
            
        if self.mano == objetivo_key:
            if info["proceso"] == "cocinar": return COORDENADAS_ESTACIONES["HO"]
            if info["proceso"] == "cortar": return COORDENADAS_ESTACIONES["CO"]
            return COORDENADAS_ESTACIONES["PL"]
        
        if info["proceso"] == "cocinar" and self.sarten["item"] is not None:
             return COORDENADAS_ESTACIONES["HO"]

        return COORDENADAS_ESTACIONES[info["loc"]]

    def step(self, action_idx):
        """
        FUNCIÓN DE TRANSICIÓN
        Input: Acción del agente.
        Output: Nuevo Estado, Recompensa conseguida, si ha terminado (Done).
        """
        self.steps_taken += 1
        reward = -1  # LIVING PENALTY: Restamos puntos por cada paso para incentivar rutas rápidas.
        info = ""
        
        # 1. Calcular movimiento basado en la acción
        move = [0, 0]
        if action_idx == 0: move = [-1, 0] 
        elif action_idx == 1: move = [1, 0] 
        elif action_idx == 2: move = [0, -1]
        elif action_idx == 3: move = [0, 1] 
        
        old_pos = list(self.agent_pos)
        new_r, new_c = self.agent_pos[0] + move[0], self.agent_pos[1] + move[1]
        
        # 2. Gestión de colisiones (Muros y Obstáculos)
        choque = False
        if 0 <= new_r < FILAS and 0 <= new_c < COLS:
            celda = MAPA[new_r][new_c]
            if celda == "O": 
                reward = -5  # Penalización fuerte por chocar
                choque = True
            elif celda != "P": 
                pass # No camina sobre las mesas
            else:
                self.agent_pos = [new_r, new_c]
        else:
            choque = True
            reward = -5

        # 3. Aplicar Reward Shaping (Recompensa por acercarse)
        if not choque:
            target_pos = self.get_target_pos()
            # Distancia Manhattan (bloques)
            dist_old = abs(old_pos[0] - target_pos[0]) + abs(old_pos[1] - target_pos[1])
            dist_new = abs(self.agent_pos[0] - target_pos[0]) + abs(self.agent_pos[1] - target_pos[1])
            if dist_new < dist_old: reward += 1 # ¡Bien hecho! Te has acercado.

        # 4. Actualizar estado de la sartén (independiente del agente)
        if self.sarten["item"] is not None:
            self.sarten["tiempo"] += 1
            if self.sarten["tiempo"] == 6:
                reward -= 50 # Castigo: Se ha quemado la comida

        # 5. Ejecutar Interacción si la acción es "INTERACTUAR"
        if action_idx == 4:
            reward += self._handle_interaction()

        # 6. Comprobar si se ha acabado el tiempo
        if self.steps_taken >= self.max_steps:
            self.done = True
            
        return self.get_state(), reward, self.done, info

    def _avanzar_paso(self):
        self.paso_actual += 1

    def _handle_interaction(self):
        """Lógica detallada de interacción con las estaciones y asignación de recompensas específicas."""
        # Detectar qué estación tiene delante
        estaciones_adyacentes = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = self.agent_pos[0]+dr, self.agent_pos[1]+dc
            if 0 <= nr < FILAS and 0 <= nc < COLS:
                celda = MAPA[nr][nc]
                if celda not in ["P", "O"]:
                    estaciones_adyacentes.append(celda)
        
        if not estaciones_adyacentes: return -0.5 # Penalización pequeña por interactuar con el aire

        estacion = estaciones_adyacentes[0] 
        objetivo_key = self.receta[self.paso_actual] if self.paso_actual < len(self.receta) else None
        
        # --- LÓGICA DE AUTO-CORRECCIÓN (Para facilitar el entrenamiento) ---
        # Si tiene basura, asumimos que quiere usar la papelera (TR)
        if "TR" in estaciones_adyacentes and (self.mano == "BASURA" or self._item_en_mano_es_inutil()):
            estacion = "TR"
        # Si tiene carne cruda, asumimos que quiere usar la sartén (HO)
        elif "HO" in estaciones_adyacentes and self.mano == objetivo_key and objetivo_key in ["carne", "bacon"]:
            estacion = "HO"
        # Etcétera para las otras estaciones...
        elif "CO" in estaciones_adyacentes and self.mano == objetivo_key:
            info = INFO_INGREDIENTES.get(objetivo_key)
            if info and info["proceso"] == "cortar":
                estacion = "CO"
        elif "EN" in estaciones_adyacentes and self.mano == "plato_lleno":
            estacion = "EN"
        elif "HO" in estaciones_adyacentes and self.sarten["item"] is not None and self.mano is None:
             estacion = "HO"

        # --- INTERACCIONES ESPECÍFICAS ---
        
        # 1. BASURA
        if estacion == "TR" and self.mano is not None:
            if self.mano == "plato_lleno":
                self.mano = None
                self.plato_contenidos = [] 
                self.paso_actual = 0        
                return -500 # Castigo MUY severo por tirar el plato hecho
            es_util = not self._item_en_mano_es_inutil()
            self.mano = None
            if es_util: return -50 # Castigo por tirar comida buena
            return -2 # Coste pequeño por limpiar basura

        if not objetivo_key: return 0
        info_ing = INFO_INGREDIENTES[objetivo_key]

        # 2. ENTREGA (Final del nivel)
        if objetivo_key == "entrega":
            if estacion == "EN" and self.mano == "plato_lleno":
                reward_entrega = 200 # ¡Premio grande!
                if self.nivel_actual_episodio + 1 < self.meta_hamburguesas:
                    # Si faltan hamburguesas, avanzamos nivel interno
                    self.nivel_actual_episodio += 1
                    self.receta = self.generar_receta(self.nivel_actual_episodio)
                    self.paso_actual = 0
                    self.mano = None
                    self.plato_contenidos = [] 
                    return reward_entrega
                else:
                    self.done = True # Fin del episodio
                    return reward_entrega * 2 # Bonus final
            return -1

        # 3. RECOGER PLATO
        if objetivo_key == "recoger_plato":
            if estacion == "PL" and self.mano is None:
                self.mano = "plato_lleno"
                self.plato_contenidos = [] 
                self._avanzar_paso()
                return 50
            return -1

        item_final = info_ing["final"]

        # 4. EMPLATAR (Poner ingrediente en el plato)
        if estacion == "PL":
            if self.mano == item_final:
                self.plato_contenidos.append(self.mano)
                self.mano = None
                self._avanzar_paso()
                return 60 # Premio por completar un paso de la receta
            elif self.mano is not None:
                return -5

        # 5. COGER INGREDIENTE (Despensa)
        if estacion == info_ing["loc"] and self.mano is None:
            self.mano = objetivo_key
            return 0 

        # 6. CORTAR (Mesa de corte)
        if estacion == "CO" and self.mano == objetivo_key and info_ing["proceso"] == "cortar":
            self.mano = f"{self.mano}_cortado"
            return 30

        # 7. COCINAR (Sartén)
        if estacion == "HO":
            # Poner en la sartén
            if self.mano == objetivo_key and info_ing["proceso"] == "cocinar" and self.sarten["item"] is None:
                self.sarten["item"] = self.mano
                self.sarten["tiempo"] = 0
                self.mano = None
                return 0
            
            # Sacar de la sartén
            elif self.sarten["item"] is not None and self.mano is None:
                tiempo = self.sarten["tiempo"]
                item = self.sarten["item"]
                self.sarten["item"] = None
                self.sarten["tiempo"] = 0
                if tiempo < 2:
                    self.mano = "BASURA" # Sacado demasiado pronto (Crudo)
                    return -50 
                elif tiempo > 5:
                    self.mano = "BASURA" # Sacado demasiado tarde (Quemado)
                    return 0 
                else:
                    self.mano = f"{item}_cocinado" # ¡Perfecto!
                    return 30 

        return -1

# =============================================================================
# 4. CLASE DEL AGENTE (Q-LEARNING AGENT)
# =============================================================================
class QAgent:
    """
    El cerebro de la IA. Utiliza Q-Learning Tabular.
    """
    def __init__(self, action_space_n):
        self.q_table = {} # Diccionario donde guardamos el conocimiento: {(estado, acción): valor}
        
        # --- HIPERPARÁMETROS DEL APRENDIZAJE POR REFUERZO ---
        self.alpha = 0.2    # Tasa de Aprendizaje (Learning Rate): Velocidad a la que sobreescribe memoria antigua.
        self.gamma = 0.95   # Factor de Descuento (Discount): Importancia de las recompensas futuras.
        self.epsilon = 1.0  # Exploración (Epsilon): 1.0 significa 100% aleatorio al inicio.
        self.epsilon_min = 0.05 # Mínimo de exploración (siempre guarda un 5% de curiosidad).
        self.epsilon_decay = 0.9995 # Velocidad a la que reduce la exploración.
        self.actions = list(range(action_space_n))

    def get_q(self, state, action):
        """Devuelve el valor Q para un estado y acción. Si no existe, devuelve 0."""
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        """
        Política Epsilon-Greedy:
        - A veces explora (aleatorio).
        - A veces explota (usa lo aprendido).
        """
        # 1. Tiramos un dado. Si es menor que epsilon, actuamos al azar (EXPLORACIÓN).
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions) 
        
        # 2. Si no, elegimos la mejor acción conocida (EXPLOTACIÓN).
        q_values = [self.get_q(state, a) for a in self.actions]
        max_q = max(q_values)
        # Si hay varias acciones igual de buenas, elegimos una al azar para no bloquearnos.
        actions_with_max_q = [a for a, q in enumerate(q_values) if q == max_q]
        return random.choice(actions_with_max_q)

    def learn(self, state, action, reward, next_state, done):
        """
        PASO DE APRENDIZAJE: ECUACIÓN DE BELLMAN.
        Actualiza el valor de la acción tomada basándose en la recompensa recibida y el futuro esperado.
        """
        old_q = self.get_q(state, action)
        
        if done:
            target = reward # Si el episodio acaba, no hay futuro.
        else:
            # El valor objetivo es: Recompensa inmediata + Valor del mejor estado futuro
            next_max = max([self.get_q(next_state, a) for a in self.actions])
            target = reward + self.gamma * next_max
        
        # Actualizamos la celda de la tabla Q acercándonos al objetivo
        self.q_table[(state, action)] = old_q + self.alpha * (target - old_q)

    def decay_epsilon(self):
        """Reduce la aleatoriedad (epsilon) después de cada episodio."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

# =============================================================================
# 5. FUNCIONES DE DIBUJADO (VISUALIZACIÓN)
# =============================================================================
def draw_game(screen, font, env, ep, total_rew):
    """Pinta el estado del juego en la ventana de Pygame."""
    screen.fill((255, 252, 242)) # Color de fondo general

    # Dibujar el borde del tablero
    offset_x = MARGEN_TABLERO 
    offset_y = MARGEN_TABLERO
    tablero_bg_rect_outer = pygame.Rect(
        offset_x - 6, offset_y - 6,
        COLS * ANCHO_CELDA + 12, FILAS * ANCHO_CELDA + 12
    )
    pygame.draw.rect(screen, MARRON_OBSTACULO, tablero_bg_rect_outer, 4)
    
    # Dibujar las celdas del grid
    for r in range(FILAS):
        for c in range(COLS):
            x, y = c * ANCHO_CELDA + offset_x, r * ANCHO_CELDA + offset_y
            rect = pygame.Rect(x, y, ANCHO_CELDA, ANCHO_CELDA)
            tipo = MAPA[r][c]
            
            # Color del suelo
            relleno_color = BLANCO
            if tipo == "P": relleno_color = MARRON_SUELO 
            elif tipo == "O": relleno_color = MARRON_OBSTACULO 
            
            pygame.draw.rect(screen, relleno_color, rect)
            
            # Debug: Resaltar en verde a dónde quiere ir la IA (Target)
            target = env.get_target_pos()
            if (r,c) == target:
                pygame.draw.rect(screen, VERDE, rect) 
            
            pygame.draw.rect(screen, (255, 252, 242), rect, 2)

            # Barra de progreso de la sartén
            if tipo == "HO":
                if env.sarten["item"]:
                    t = env.sarten["tiempo"]
                    color_barra = ROJO
                    if t >= 2 and t <= 5: color_barra = AMARILLO 
                    elif t > 5: color_barra = (50,0,0) 
                    
                    ancho_barra = min((t / 5) * ANCHO_CELDA, ANCHO_CELDA)
                    pygame.draw.rect(screen, color_barra, (x, y+ANCHO_CELDA-10, ancho_barra, 5))

            # Contenido visual del plato
            if tipo == "PL":
                if len(env.plato_contenidos) > 0:
                    small_f = pygame.font.SysFont("segoeuiemoji", 20)
                    offset_emoji = 0
                    for item in env.plato_contenidos:
                        icon = ""
                        if "pan" in item: icon = "🍞"
                        elif "carne" in item: icon = "🥩"
                        elif "queso" in item: icon = "🧀"
                        elif "tomate" in item: icon = "🍅"
                        elif "lechuga" in item: icon = "🥬"
                        elif "bacon" in item: icon = "🥓"
                        
                        screen.blit(small_f.render(icon, True, NEGRO), (x + offset_emoji, y)) 
                        offset_emoji += 15

            # Emojis de las estaciones
            txt = EMOJIS.get(tipo, "")
            if txt:
                color_texto = NEGRO if tipo not in ["P", "O", "TR"] else BLANCO 
                surf = font.render(txt, True, color_texto)
                text_rect = surf.get_rect()
                text_rect.center = (x + ANCHO_CELDA // 2, y + ANCHO_CELDA // 2)
                screen.blit(surf, text_rect)
            
    # Dibujar al Pollito Agente
    px, py = env.agent_pos[1] * ANCHO_CELDA + offset_x, env.agent_pos[0] * ANCHO_CELDA + offset_y
    icono = EMOJIS["EXITO"] if env.done else EMOJIS["POLLITO"]
    
    pollito_surf = font.render(icono, True, BLANCO)
    pollito_rect = pollito_surf.get_rect(center=(px + ANCHO_CELDA // 2, py + ANCHO_CELDA // 2))
    screen.blit(pollito_surf, pollito_rect)

    # Dibujar lo que lleva en la mano
    if env.mano:
        mano_txt = "❓"
        if "pan" in env.mano: mano_txt = "🍞"
        if "carne" in env.mano: mano_txt = "🥩"
        if "queso" in env.mano: mano_txt = "🧀"
        if "tomate" in env.mano: mano_txt = "🍅"
        if "lechuga" in env.mano: mano_txt = "🥬"
        if "bacon" in env.mano: mano_txt = "🥓"
        if "plato_lleno" in env.mano: mano_txt = "🍔"
        if "BASURA" in env.mano: mano_txt = "🤢"
        
        if "cortado" in env.mano: mano_txt += "🔪"
        if "cocinado" in env.mano: mano_txt += "♨️"
        
        small_font = pygame.font.SysFont("segoeuiemoji", 20)
        screen.blit(small_font.render(mano_txt, True, BLANCO), (px + 30, py - 5))

    # --- PANEL DE INFORMACIÓN ---
    y_separator = offset_y + FILAS * ANCHO_CELDA
    y_base = y_separator + 20
    x_base = MARGEN_TABLERO + 10
    font_s = pygame.font.SysFont("montserratblack", 18)
    
    # Construcción de la visualización de la receta (con [OK])
    receta_display = []
    for step in env.receta:
        nombre = NOMBRES_COMPLETOS.get(step, step.upper())
        receta_display.append(nombre)
    
    curr_step_idx = env.paso_actual
    receta_str = ""
    for i, r in enumerate(receta_display):
        if i == curr_step_idx:
            receta_str += f"[{r}] " # Paso actual entre corchetes
        elif i < curr_step_idx:
            receta_str += f"[OK] "  # Pasos completados
        else:
            receta_str += f"{r} "   # Pasos futuros

    pasos_restantes = env.max_steps - env.steps_taken
    
    info_strs = [
        f"EPISODIO: {ep} | REWARD: {total_rew:.0f}",
        f"META: {env.meta_hamburguesas} Hamburguesa(s)",
        f"PROGRESO: {env.nivel_actual_episodio + 1}/{env.meta_hamburguesas} (Pasos Restantes: {pasos_restantes})",
        f"RECETA ACTUAL: {receta_str}",
    ]
    
    for i, t in enumerate(info_strs):
        screen.blit(font_s.render(t, True, MARRON_OBSTACULO), (x_base, y_base + i*25))

# =============================================================================
# 6. MAIN LOOP DE ENTRENAMIENTO
# =============================================================================
def main():
    pygame.init()
    try: font = pygame.font.SysFont("segoeuiemoji", 40)
    except: font = pygame.font.SysFont("arial", 40)

    screen = pygame.display.set_mode((ANCHO_VENTANA, ALTO_VENTANA))
    pygame.display.set_caption("Pollito Chef - Aprendizaje por Refuerzo")
    clock = pygame.time.Clock()
    
    env = Environment()
    agent = QAgent(env.action_space_n)
    
    num_episodes = 5000
    victorias_consecutivas = 0
    rewards_history = []
    level_change_episodes = [0] # Guardamos el inicio como primer nivel
    
    prev_meta = env.meta_hamburguesas

    # --- BUCLE PRINCIPAL DE ENTRENAMIENTO ---
    for ep in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        # En el último episodio desactivamos la exploración para evaluar
        es_ultimo_tramo = (ep >= num_episodes - 1)
        if es_ultimo_tramo:
            agent.epsilon = 0.0

        # Control de renderizado: No dibujamos todos los frames para ir rápido
        render = (ep < 3) or (ep % 500 == 0) or es_ultimo_tramo
        
        while not done:
            if render:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT: 
                        pygame.quit()
                        return

            # 1. ELEGIR ACCIÓN
            action = agent.choose_action(state)
            
            # 2. EJECUTAR ACCIÓN EN EL ENTORNO
            next_state, reward, done, info = env.step(action)
            
            # 3. APRENDER (Actualizar Q-Table)
            agent.learn(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
            
            if render:
                draw_game(screen, font, env, ep, total_reward)
                pygame.display.flip()
                
                if es_ultimo_tramo:
                    clock.tick(10) # Velocidad normal para ver el resultado final
                else:
                    clock.tick(FPS) # Velocidad máxima entrenando

        # Decaimiento de epsilon al final de cada episodio
        agent.decay_epsilon()
        rewards_history.append(total_reward)
        
        # --- LÓGICA DE SUBIDA DE NIVEL (Curriculum Learning) ---
        gano_episodio = (done and env.nivel_actual_episodio == env.meta_hamburguesas - 1)
        
        if gano_episodio:
            victorias_consecutivas += 1
            if victorias_consecutivas >= 5:
                if env.meta_hamburguesas < 5:
                    env.meta_hamburguesas += 1
                    victorias_consecutivas = 0
                    print(f"¡NIVEL GLOBAL SUBIDO! Ahora debe hacer {env.meta_hamburguesas} hamburguesas.")
        else:
            victorias_consecutivas = 0
        
        # Detectar cambio de nivel para la gráfica
        if env.meta_hamburguesas > prev_meta:
            level_change_episodes.append(ep)
            prev_meta = env.meta_hamburguesas

        if ep < 3 or ep >= num_episodes - 1 or ep % 250 == 0:
            print(f"Ep: {ep} | Rwd: {total_reward:.1f} | Meta: {env.meta_hamburguesas}")

    # Añadimos el final para el último rango
    level_change_episodes.append(num_episodes)

    print("¡Entrenamiento completado!")
    pygame.quit()
    
    # =============================================================================
    # 8. GENERACIÓN DE GRÁFICAS
    # =============================================================================
    plt.figure(figsize=(12, 6))
    
    # Datos crudos
    plt.plot(rewards_history, label="Reward por Episodio", alpha=0.3, color="gray")
    
    # Media Móvil (para ver la tendencia limpia)
    window_size = 50
    if len(rewards_history) >= window_size:
        moving_avg = np.convolve(rewards_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(rewards_history)), moving_avg, color="orange", label=f"Media Móvil ({window_size})", linewidth=2)

    # Líneas verticales de cambio de nivel
    y_bottom, y_top = plt.ylim()
    range_y = y_top - y_bottom
    
    for i in range(len(level_change_episodes) - 1):
        start = level_change_episodes[i]
        end = level_change_episodes[i+1]
        
        if i > 0: 
            plt.axvline(x=start, color='black', linestyle='--', alpha=0.7)
            plt.text(start, y_top, f"Nivel {i+1}", rotation=90, verticalalignment='top')

        # Estadísticas por tramo
        segment = rewards_history[start:end]
        if len(segment) > 0:
            seg_avg = sum(segment) / len(segment)
            mid_x = start + (end - start) / 2
            
            # Alternar posición del texto para que no se solapen
            if i % 2 == 0: text_y_pos = y_bottom + range_y * 0.50 
            else: text_y_pos = y_bottom + range_y * 0.10 
            
            stats_text = f"Nivel {i+1}\nProm:{seg_avg:.0f}"
            plt.text(mid_x, text_y_pos, stats_text, 
                     bbox=dict(facecolor='white', alpha=0.9, edgecolor='black'),
                     horizontalalignment='center', fontsize=8)

    plt.title("Evolución del Aprendizaje de Pollito Chef (Por Niveles)")
    plt.xlabel("Episodio")
    plt.ylabel("Reward Total")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()