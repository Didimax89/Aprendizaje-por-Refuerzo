<div align="center">
  <h1>Aprendizaje por Refuerzo</h1>
  <p><em>Práctica desarrollada para la asignatura <b>Inteligencia Artificial I</b> (Grado Ingeniería Informática UCM)</em></p>

  <p>
    <img src="https://img.shields.io/badge/Nota_Práctica-10%2F10-success" alt="Nota" />
    <img src="https://img.shields.io/badge/Concurso-2%C2%BA_Puesto-purple" alt="Top 2" />
    <img src="https://img.shields.io/badge/Nota_Asignatura-Matr%C3%ADcula_de_Honor-gold" alt="MH" />
  </p>
</div>

---

Este repositorio contiene la implementación de un agente inteligente que aprende desde cero a operar en una cocina dinámica inspirada en el videojuego *Overcooked*. El agente (un pollito), mediante un proceso de ensayo y error, descubre de forma autónoma cómo navegar, procesar ingredientes en distintas estaciones y gestionar los tiempos de cocción para completar recetas de hamburguesas.

### Algunos conceptos clave
* Q-Learning
* Curriculum Learning
* Reward Shaping
* Visualización con Pygame

Para ver toda la explicación técnica, el modelado (MDP) y el análisis de resultados, consultar el **[Informe de la práctica (PDF)](informe_lab6_laura_diego.pdf)**.

---

### Ejecución
```bash
# 1. Instalar dependencias
pip install pygame numpy matplotlib

# 2. Ejecutar la simulación
python codigo_lab6_laura_diego.py
