"""
Movimiento Browniano en un Campo Magnético
Simulación de la Ecuación de Langevin Tridimensional

Procesos Estocásticos - Problema 3
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy import stats

class BrownianMagneticField:
    """
    Simulador de movimiento browniano en un campo magnético uniforme
    """
    
    def __init__(self, m=1.0, q=1.0, beta=1.0, B=1.0, kB=1.0, T=1.0):
        """
        Parámetros:
        -----------
        m : float
            Masa de la partícula
        q : float
            Carga de la partícula
        beta : float
            Coeficiente de fricción
        B : float
            Magnitud del campo magnético (dirección +z)
        kB : float
            Constante de Boltzmann
        T : float
            Temperatura
        """
        self.m = m
        self.q = q
        self.beta = beta
        self.B = B
        self.kB = kB
        self.T = T
        
        # Parámetros derivados
        self.gamma = beta / m  # Coeficiente de amortiguamiento
        self.omega_c = q * B / m  # Frecuencia ciclotrónica
        self.sigma_noise = np.sqrt(2 * beta * kB * T / (m * m))  # Intensidad del ruido
        
        # Para comparación
        self.v_thermal = np.sqrt(kB * T / m)  # Velocidad térmica típica
        
    def info(self):
        """Imprime información sobre los parámetros del sistema"""
        print("="*60)
        print("PARÁMETROS DEL SISTEMA")
        print("="*60)
        print(f"Masa (m):                    {self.m:.3e}")
        print(f"Carga (q):                   {self.q:.3e}")
        print(f"Fricción (β):                {self.beta:.3e}")
        print(f"Campo magnético (B):         {self.B:.3e}")
        print(f"Temperatura (T):             {self.T:.3e}")
        print(f"kB:                          {self.kB:.3e}")
        print("-"*60)
        print("PARÁMETROS DERIVADOS")
        print("-"*60)
        print(f"Amortiguamiento (γ = β/m):   {self.gamma:.3e} s⁻¹")
        print(f"Frecuencia ciclotrónica (ωc):{self.omega_c:.3e} s⁻¹")
        print(f"Relación ωc/γ:               {self.omega_c/self.gamma:.3e}")
        print(f"Velocidad térmica:           {self.v_thermal:.3e}")
        print(f"Intensidad ruido:            {self.sigma_noise:.3e}")
        print(f"Tiempo de relajación (1/γ):  {1/self.gamma:.3e} s")
        if self.omega_c > 0:
            print(f"Período ciclotrónico (2π/ωc):{2*np.pi/self.omega_c:.3e} s")
        print("="*60)
        
        # Clasificar el régimen
        if self.omega_c < 0.1 * self.gamma:
            print("RÉGIMEN: Campo magnético DÉBIL (ωc ≪ γ)")
        elif self.omega_c < self.gamma:
            print("RÉGIMEN: Campo magnético MODERADO (ωc < γ)")
        elif self.omega_c < 10 * self.gamma:
            print("RÉGIMEN: Campo magnético FUERTE (ωc > γ)")
        else:
            print("RÉGIMEN: Campo magnético MUY FUERTE (ωc ≫ γ)")
        print("="*60 + "\n")
    
    def simulate(self, V0, t_max, dt, seed=None):
        """
        Simula la dinámica de la velocidad usando Euler-Maruyama
        
        Parámetros:
        -----------
        V0 : array_like, shape (3,)
            Velocidad inicial [Vx, Vy, Vz]
        t_max : float
            Tiempo máximo de simulación
        dt : float
            Paso de tiempo
        seed : int, optional
            Semilla para reproducibilidad
        
        Retorna:
        --------
        t : ndarray
            Array de tiempos
        V : ndarray, shape (N, 3)
            Trayectoria de velocidades [Vx, Vy, Vz]
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Número de pasos
        n_steps = int(t_max / dt)
        
        # Arrays para guardar resultados
        t = np.linspace(0, t_max, n_steps)
        V = np.zeros((n_steps, 3))
        V[0] = V0
        
        # Prefactor para el ruido
        noise_amplitude = self.sigma_noise * np.sqrt(dt)
        
        # Integración de Euler-Maruyama
        for i in range(n_steps - 1):
            Vx, Vy, Vz = V[i]
            
            # Términos deterministas
            dVx_det = (-self.gamma * Vx + self.omega_c * Vy) * dt
            dVy_det = (-self.gamma * Vy - self.omega_c * Vx) * dt
            dVz_det = -self.gamma * Vz * dt
            
            # Términos estocásticos
            xi = np.random.randn(3)  # Ruido gaussiano blanco
            dVx_stoch = noise_amplitude * xi[0]
            dVy_stoch = noise_amplitude * xi[1]
            dVz_stoch = noise_amplitude * xi[2]
            
            # Actualización
            V[i+1, 0] = Vx + dVx_det + dVx_stoch
            V[i+1, 1] = Vy + dVy_det + dVy_stoch
            V[i+1, 2] = Vz + dVz_det + dVz_stoch
        
        return t, V


def plot_velocity_components(t, V, B_value, save_name=None):
    """
    Grafica las tres componentes de la velocidad vs tiempo
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    components = ['x', 'y', 'z']
    colors = ['blue', 'red', 'green']
    
    for i, (ax, comp, color) in enumerate(zip(axes, components, colors)):
        ax.plot(t, V[:, i], color=color, linewidth=1, alpha=0.8)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=0.5, alpha=0.5)
        ax.set_ylabel(f'$V_{comp}(t)$', fontsize=13)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(t[0], t[-1])
    
    axes[0].set_title(f'Componentes de Velocidad (B = {B_value})', 
                     fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Tiempo $t$', fontsize=13)
    
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_name}")
    plt.show()


def plot_velocity_magnitude(t, V, B_value, kBT_over_m, save_name=None):
    """
    Grafica la magnitud de la velocidad y velocidad perpendicular
    """
    V_mag = np.sqrt(V[:, 0]**2 + V[:, 1]**2 + V[:, 2]**2)
    V_perp = np.sqrt(V[:, 0]**2 + V[:, 1]**2)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Magnitud total
    axes[0].plot(t, V_mag, 'purple', linewidth=1, alpha=0.8, label='$|V(t)|$')
    v_eq = np.sqrt(3 * kBT_over_m)
    axes[0].axhline(y=v_eq, color='red', linestyle='--', linewidth=2, 
                   label=f'Equilibrio: $\sqrt{{3k_BT/m}}$ = {v_eq:.2f}')
    axes[0].set_ylabel('$|V(t)|$', fontsize=13)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f'Magnitudes de Velocidad (B = {B_value})', 
                     fontsize=14, fontweight='bold')
    
    # Velocidad perpendicular
    axes[1].plot(t, V_perp, 'orange', linewidth=1, alpha=0.8, label='$V_\\perp(t)$')
    v_perp_eq = np.sqrt(2 * kBT_over_m)
    axes[1].axhline(y=v_perp_eq, color='red', linestyle='--', linewidth=2,
                   label=f'Equilibrio: $\sqrt{{2k_BT/m}}$ = {v_perp_eq:.2f}')
    axes[1].set_xlabel('Tiempo $t$', fontsize=13)
    axes[1].set_ylabel('$V_\\perp(t)$', fontsize=13)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_name}")
    plt.show()


def plot_phase_space_2D(V, B_value, kBT_over_m, save_name=None):
    """
    Grafica la trayectoria en el espacio de fases (Vx, Vy)
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Trayectoria
    ax.plot(V[:, 0], V[:, 1], 'b-', linewidth=0.5, alpha=0.6)
    ax.plot(V[0, 0], V[0, 1], 'go', markersize=10, label='Inicio', zorder=5)
    ax.plot(V[-1, 0], V[-1, 1], 'ro', markersize=10, label='Final', zorder=5)
    
    # Círculo de equilibrio térmico
    theta = np.linspace(0, 2*np.pi, 100)
    v_eq = np.sqrt(2 * kBT_over_m)
    ax.plot(v_eq * np.cos(theta), v_eq * np.sin(theta), 'r--', 
           linewidth=2, label=f'Equilibrio: $\sqrt{{2k_BT/m}}$ = {v_eq:.2f}')
    
    ax.set_xlabel('$V_x$', fontsize=14)
    ax.set_ylabel('$V_y$', fontsize=14)
    ax.set_title(f'Espacio de Fases en el Plano xy (B = {B_value})', 
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Límites basados en el equilibrio
    lim = 3 * v_eq
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_name}")
    plt.show()


def plot_phase_space_3D(V, B_value, save_name=None):
    """
    Grafica la trayectoria en el espacio de fases 3D (Vx, Vy, Vz)
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Trayectoria completa
    ax.plot(V[:, 0], V[:, 1], V[:, 2], 'b-', linewidth=0.5, alpha=0.6)
    
    # Puntos inicial y final
    ax.scatter(V[0, 0], V[0, 1], V[0, 2], c='green', s=100, 
              marker='o', label='Inicio', zorder=5)
    ax.scatter(V[-1, 0], V[-1, 1], V[-1, 2], c='red', s=100, 
              marker='o', label='Final', zorder=5)
    
    ax.set_xlabel('$V_x$', fontsize=13)
    ax.set_ylabel('$V_y$', fontsize=13)
    ax.set_zlabel('$V_z$', fontsize=13)
    ax.set_title(f'Trayectoria 3D en Espacio de Velocidades (B = {B_value})', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_name}")
    plt.show()


def plot_angle_evolution(t, V, omega_c, save_name=None):
    """
    Grafica la evolución del ángulo en el plano xy
    """
    # Calcular ángulo
    theta = np.arctan2(V[:, 1], V[:, 0])
    
    # Unwrap para evitar discontinuidades
    theta_unwrap = np.unwrap(theta)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(t, theta_unwrap, 'b-', linewidth=1.5, label='$\\theta(t)$')
    
    # Predicción teórica: rotación con frecuencia -omega_c
    if omega_c > 0:
        theta_theory = theta_unwrap[0] - omega_c * t
        ax.plot(t, theta_theory, 'r--', linewidth=2, alpha=0.7,
               label=f'Teórico: $\\theta_0 - \\omega_c t$ ($\\omega_c$ = {omega_c:.2f})')
    
    ax.set_xlabel('Tiempo $t$', fontsize=13)
    ax.set_ylabel('Ángulo $\\theta$ [rad]', fontsize=13)
    ax.set_title('Evolución del Ángulo en el Plano xy', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_name}")
    plt.show()


def analyze_equilibrium_statistics(V, kBT_over_m, skip_initial=0.3):
    """
    Analiza las estadísticas de equilibrio
    
    Parámetros:
    -----------
    V : ndarray
        Trayectoria de velocidades
    kBT_over_m : float
        Valor teórico de kBT/m
    skip_initial : float
        Fracción de la trayectoria a descartar (termalización)
    """
    # Descartar parte inicial (termalización)
    n_skip = int(len(V) * skip_initial)
    V_eq = V[n_skip:]
    
    print("\n" + "="*60)
    print("ESTADÍSTICAS DE EQUILIBRIO")
    print("="*60)
    print(f"Muestras analizadas: {len(V_eq)}")
    print(f"Valor teórico kBT/m: {kBT_over_m:.4f}")
    print("-"*60)
    
    # Promedios
    mean_Vx = np.mean(V_eq[:, 0])
    mean_Vy = np.mean(V_eq[:, 1])
    mean_Vz = np.mean(V_eq[:, 2])
    
    print("VALORES ESPERADOS (deben ser ≈ 0):")
    print(f"  <Vx> = {mean_Vx:+.4f}")
    print(f"  <Vy> = {mean_Vy:+.4f}")
    print(f"  <Vz> = {mean_Vz:+.4f}")
    print("-"*60)
    
    # Varianzas
    var_Vx = np.var(V_eq[:, 0])
    var_Vy = np.var(V_eq[:, 1])
    var_Vz = np.var(V_eq[:, 2])
    
    print(f"VARIANZAS (deben ser ≈ {kBT_over_m:.4f}):")
    print(f"  <Vx²> = {var_Vx:.4f}  (error: {abs(var_Vx - kBT_over_m)/kBT_over_m * 100:.1f}%)")
    print(f"  <Vy²> = {var_Vy:.4f}  (error: {abs(var_Vy - kBT_over_m)/kBT_over_m * 100:.1f}%)")
    print(f"  <Vz²> = {var_Vz:.4f}  (error: {abs(var_Vz - kBT_over_m)/kBT_over_m * 100:.1f}%)")
    print("-"*60)
    
    # Energía cinética promedio
    E_kin = 0.5 * np.sum(V_eq**2, axis=1)
    mean_E = np.mean(E_kin)
    theory_E = 1.5 * kBT_over_m  # (3/2) kBT / m, con m=1
    
    print(f"ENERGÍA CINÉTICA:")
    print(f"  <E> = {mean_E:.4f}")
    print(f"  Teórico (3/2 kBT/m) = {theory_E:.4f}")
    print(f"  Error: {abs(mean_E - theory_E)/theory_E * 100:.1f}%")
    print("-"*60)
    
    # Correlación cruzada
    corr_xy = np.mean(V_eq[:, 0] * V_eq[:, 1])
    corr_xz = np.mean(V_eq[:, 0] * V_eq[:, 2])
    corr_yz = np.mean(V_eq[:, 1] * V_eq[:, 2])
    
    print("CORRELACIONES CRUZADAS (deben ser ≈ 0):")
    print(f"  <Vx·Vy> = {corr_xy:+.4f}")
    print(f"  <Vx·Vz> = {corr_xz:+.4f}")
    print(f"  <Vy·Vz> = {corr_yz:+.4f}")
    print("="*60 + "\n")


def plot_velocity_histograms(V, kBT_over_m, skip_initial=0.3, save_name=None):
    """
    Grafica histogramas de las componentes de velocidad
    """
    n_skip = int(len(V) * skip_initial)
    V_eq = V[n_skip:]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    components = ['x', 'y', 'z']
    colors = ['blue', 'red', 'green']
    
    # Distribución teórica
    sigma = np.sqrt(kBT_over_m)
    v_range = np.linspace(-4*sigma, 4*sigma, 100)
    gaussian = stats.norm.pdf(v_range, 0, sigma)
    
    for i, (ax, comp, color) in enumerate(zip(axes, components, colors)):
        # Histograma
        ax.hist(V_eq[:, i], bins=50, density=True, alpha=0.6, 
               color=color, edgecolor='black', label='Simulación')
        
        # Distribución teórica
        ax.plot(v_range, gaussian, 'k--', linewidth=2.5, 
               label=f'Teórico: $\mathcal{{N}}(0, {sigma:.2f})$')
        
        ax.set_xlabel(f'$V_{comp}$', fontsize=13)
        ax.set_ylabel('Densidad de probabilidad', fontsize=13)
        ax.set_title(f'Componente ${comp}$', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Distribuciones de Velocidad en Equilibrio', 
                fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_name:
        plt.savefig(save_name, dpi=300, bbox_inches='tight')
        print(f"Figura guardada: {save_name}")
    plt.show()


def compare_different_B_fields():
    """
    Compara el comportamiento para diferentes intensidades de campo magnético
    """
    # Parámetros fijos
    m, q, beta, kB, T = 1.0, 1.0, 1.0, 1.0, 1.0
    B_values = [0.0, 0.5, 1.0, 2.0, 5.0]
    
    # Condición inicial
    V0 = np.array([2.0, 0.0, 1.0])
    
    # Parámetros de simulación
    t_max = 50.0
    dt = 0.01
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for idx, B in enumerate(B_values[:5]):
        simulator = BrownianMagneticField(m, q, beta, B, kB, T)
        t, V = simulator.simulate(V0, t_max, dt, seed=42)
        
        ax = axes[idx]
        
        # Trayectoria en plano xy
        ax.plot(V[:, 0], V[:, 1], 'b-', linewidth=0.8, alpha=0.7)
        ax.plot(V[0, 0], V[0, 1], 'go', markersize=8, label='Inicio')
        ax.plot(V[-1, 0], V[-1, 1], 'ro', markersize=8, label='Final')
        
        # Círculo de equilibrio
        v_eq = np.sqrt(2 * kB * T / m)
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(v_eq * np.cos(theta), v_eq * np.sin(theta), 'r--', 
               linewidth=1.5, alpha=0.5)
        
        omega_c = simulator.omega_c
        ax.set_title(f'B = {B:.1f}, $\\omega_c$ = {omega_c:.2f}, $\\omega_c/\\gamma$ = {omega_c/simulator.gamma:.2f}',
                    fontsize=11, fontweight='bold')
        ax.set_xlabel('$V_x$', fontsize=11)
        ax.set_ylabel('$V_y$', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        ax.legend(fontsize=9)
        
        lim = 3.5
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    
    # Ocultar el último subplot
    axes[5].set_visible(False)
    
    plt.suptitle('Comparación de Trayectorias para Diferentes Campos Magnéticos', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('brownian_B_comparison.png', dpi=300, bbox_inches='tight')
    print("Figura guardada: brownian_B_comparison.png")
    plt.show()


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("MOVIMIENTO BROWNIANO EN CAMPO MAGNÉTICO")
    print("Simulación de la Ecuación de Langevin 3D")
    print("="*60 + "\n")
    
    # ==========================================================================
    # SIMULACIÓN 1: Campo magnético moderado
    # ==========================================================================
    print("\n### SIMULACIÓN 1: Campo magnético moderado (ωc ≈ γ) ###\n")
    
    # Parámetros
    m = 1.0
    q = 1.0
    beta = 1.0
    B = 1.0  # Campo moderado
    kB = 1.0
    T = 1.0
    
    # Crear simulador
    sim1 = BrownianMagneticField(m, q, beta, B, kB, T)
    sim1.info()
    
    # Condición inicial
    V0 = np.array([2.0, 0.0, 1.5])
    
    # Parámetros de simulación
    t_max = 50.0
    dt = 0.01  # Paso de tiempo suficientemente pequeño
    
    print(f"Condición inicial: V0 = [{V0[0]}, {V0[1]}, {V0[2]}]")
    print(f"Tiempo de simulación: {t_max}")
    print(f"Paso de tiempo: {dt}")
    print(f"Número de pasos: {int(t_max/dt)}\n")
    
    # Simular
    print("Ejecutando simulación...")
    t1, V1 = sim1.simulate(V0, t_max, dt, seed=42)
    print("Simulación completada.\n")
    
    # Graficar componentes
    print("Generando gráficas...")
    plot_velocity_components(t1, V1, B, 'brownian_components_B1.png')
    
    # Graficar magnitudes
    kBT_over_m = kB * T / m
    plot_velocity_magnitude(t1, V1, B, kBT_over_m, 'brownian_magnitude_B1.png')
    
    # Espacio de fases 2D
    plot_phase_space_2D(V1, B, kBT_over_m, 'brownian_phase2D_B1.png')
    
    # Espacio de fases 3D
    plot_phase_space_3D(V1, B, 'brownian_phase3D_B1.png')
    
    # Evolución del ángulo
    plot_angle_evolution(t1, V1, sim1.omega_c, 'brownian_angle_B1.png')
    
    # Histogramas
    plot_velocity_histograms(V1, kBT_over_m, save_name='brownian_histograms_B1.png')
    
    # Estadísticas
    analyze_equilibrium_statistics(V1, kBT_over_m)
    
    # ==========================================================================
    # SIMULACIÓN 2: Campo magnético fuerte
    # ==========================================================================
    print("\n### SIMULACIÓN 2: Campo magnético fuerte (ωc >> γ) ###\n")
    
    B_strong = 5.0
    sim2 = BrownianMagneticField(m, q, beta, B_strong, kB, T)
    sim2.info()
    
    t2, V2 = sim2.simulate(V0, t_max, dt, seed=42)
    
    plot_velocity_components(t2, V2, B_strong, 'brownian_components_B5.png')
    plot_phase_space_2D(V2, B_strong, kBT_over_m, 'brownian_phase2D_B5.png')
    
    # ==========================================================================
    # COMPARACIÓN DE DIFERENTES CAMPOS
    # ==========================================================================
    print("\n### COMPARACIÓN DE MÚLTIPLES CAMPOS MAGNÉTICOS ###\n")
    compare_different_B_fields()
    
    print("\n" + "="*60)
    print("SIMULACIÓN COMPLETADA")
    print("="*60)
