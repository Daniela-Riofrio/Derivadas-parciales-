import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from sympy import symbols, diff, lambdify, sympify, latex
import plotly.graph_objects as go
import plotly.express as px

# Configuración de la página
st.set_page_config(page_title="Calculadora de Derivadas Parciales", layout="wide")

# Título principal
st.title("∂ Calculadora de Derivadas Parciales")
st.markdown("Esta aplicación calcula las derivadas parciales de funciones de varias variables y muestra sus gráficos.")

# Definir variables simbólicas
x, y, z = symbols('x y z')

def calcular_derivadas_parciales(func_str, variables):
    """
    Calcula las derivadas parciales de una función
    """
    try:
        # Convertir string a expresión simbólica
        func = sympify(func_str)
        
        derivadas = {}
        for var in variables:
            derivadas[f'd/d{var}'] = diff(func, var)
        
        return func, derivadas
    except Exception as e:
        return None, str(e)

def evaluar_funcion(func, x_vals, y_vals, punto_x=None, punto_y=None):
    """
    Evalúa una función en un grid de valores
    """
    try:
        # Convertir a función numérica
        func_numeric = lambdify((x, y), func, 'numpy')
        
        if punto_x is not None and punto_y is not None:
            # Evaluar en un punto específico
            return func_numeric(punto_x, punto_y)
        else:
            # Evaluar en un grid
            X, Y = np.meshgrid(x_vals, y_vals)
            Z = func_numeric(X, Y)
            return X, Y, Z
    except Exception as e:
        return None

def crear_grafico_3d(func, x_range, y_range, derivadas=None, punto=None):
    """
    Crea gráfico 3D de la función y sus derivadas parciales
    """
    x_vals = np.linspace(x_range[0], x_range[1], 50)
    y_vals = np.linspace(y_range[0], y_range[1], 50)
    
    try:
        X, Y, Z = evaluar_funcion(func, x_vals, y_vals)
        
        # Crear figura con subplots
        fig = plt.figure(figsize=(15, 10))
        
        # Gráfico principal de la función
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        surf = ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('f(x,y)')
        ax1.set_title('Función Original f(x,y)')
        
        # Punto específico si se proporciona
        if punto:
            px, py = punto
            pz = evaluar_funcion(func, None, None, px, py)
            if pz is not None:
                ax1.scatter([px], [py], [pz], color='red', s=100, label=f'Punto ({px}, {py})')
                ax1.legend()
        
        # Gráfico de derivada parcial respecto a x
        if derivadas and 'd/dx' in derivadas:
            ax2 = fig.add_subplot(2, 2, 2, projection='3d')
            try:
                X_dx, Y_dx, Z_dx = evaluar_funcion(derivadas['d/dx'], x_vals, y_vals)
                ax2.plot_surface(X_dx, Y_dx, Z_dx, cmap='coolwarm', alpha=0.8)
                ax2.set_xlabel('X')
                ax2.set_ylabel('Y')
                ax2.set_zlabel('∂f/∂x')
                ax2.set_title('Derivada Parcial ∂f/∂x')
            except:
                ax2.text(0.5, 0.5, 0.5, 'Error al graficar', transform=ax2.transAxes)
        
        # Gráfico de derivada parcial respecto a y
        if derivadas and 'd/dy' in derivadas:
            ax3 = fig.add_subplot(2, 2, 3, projection='3d')
            try:
                X_dy, Y_dy, Z_dy = evaluar_funcion(derivadas['d/dy'], x_vals, y_vals)
                ax3.plot_surface(X_dy, Y_dy, Z_dy, cmap='plasma', alpha=0.8)
                ax3.set_xlabel('X')
                ax3.set_ylabel('Y')
                ax3.set_zlabel('∂f/∂y')
                ax3.set_title('Derivada Parcial ∂f/∂y')
            except:
                ax3.text(0.5, 0.5, 0.5, 'Error al graficar', transform=ax3.transAxes)
        
        # Gráfico de contorno
        ax4 = fig.add_subplot(2, 2, 4)
        contour = ax4.contour(X, Y, Z, levels=20, cmap='viridis')
        ax4.clabel(contour, inline=True, fontsize=8)
        ax4.set_xlabel('X')
        ax4.set_ylabel('Y')
        ax4.set_title('Curvas de Nivel')
        ax4.grid(True, alpha=0.3)
        
        # Punto en el contorno
        if punto:
            ax4.plot(punto[0], punto[1], 'ro', markersize=8, label=f'Punto ({punto[0]}, {punto[1]})')
            ax4.legend()
        
        plt.tight_layout()
        return fig
    
    except Exception as e:
        st.error(f"Error al crear el gráfico: {str(e)}")
        return None

# Sidebar para configuración
st.sidebar.header("⚙️ Configuración")

# Entrada de función
st.sidebar.subheader("📝 Función")
funcion_ejemplos = [
    "x**2 + y**2",
    "x**2 - y**2", 
    "sin(x) + cos(y)",
    "exp(x) * y",
    "x**3 + y**3 - 3*x*y",
    "log(x**2 + y**2 + 1)",
    "x*y + x**2 + y**2"
]

ejemplo_seleccionado = st.sidebar.selectbox(
    "Selecciona un ejemplo:", 
    ["Función personalizada"] + funcion_ejemplos
)

if ejemplo_seleccionado == "Función personalizada":
    func_input = st.sidebar.text_input(
        "Ingresa la función f(x,y):", 
        value="x**2 + y**2",
        help="Usa ** para potencias, sin/cos para funciones trigonométricas, exp para exponencial, log para logaritmo"
    )
else:
    func_input = ejemplo_seleccionado

# Punto específico para evaluar
st.sidebar.subheader("📍 Punto de Evaluación")
punto_x = st.sidebar.number_input("Coordenada X:", value=1.0, step=0.1)
punto_y = st.sidebar.number_input("Coordenada Y:", value=1.0, step=0.1)

# Rango de graficación
st.sidebar.subheader("📊 Rango de Graficación")
x_min = st.sidebar.number_input("X mínimo:", value=-3.0, step=0.1)
x_max = st.sidebar.number_input("X máximo:", value=3.0, step=0.1)
y_min = st.sidebar.number_input("Y mínimo:", value=-3.0, step=0.1)
y_max = st.sidebar.number_input("Y máximo:", value=3.0, step=0.1)

# Procesar función
if func_input:
    func, derivadas = calcular_derivadas_parciales(func_input, [x, y])
    
    if func is not None:
        # Mostrar función original
        st.subheader("📋 Función Original")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Función ingresada:**")
            st.code(f"f(x,y) = {func_input}")
        
        with col2:
            st.write("**Notación matemática:**")
            st.latex(f"f(x,y) = {latex(func)}")
        
        # Mostrar derivadas parciales
        st.subheader("∂ Derivadas Parciales")
        
        col3, col4 = st.columns(2)
        
        with col3:
            st.write("**Derivada parcial respecto a x:**")
            st.latex(f"\\frac{{\\partial f}}{{\\partial x}} = {latex(derivadas['d/dx'])}")
            
        with col4:
            st.write("**Derivada parcial respecto a y:**")
            st.latex(f"\\frac{{\\partial f}}{{\\partial y}} = {latex(derivadas['d/dy'])}")
        
        # Evaluar en punto específico
        st.subheader("📍 Evaluación en Punto Específico")
        
        try:
            # Evaluar función en el punto
            valor_func = evaluar_funcion(func, None, None, punto_x, punto_y)
            valor_dx = evaluar_funcion(derivadas['d/dx'], None, None, punto_x, punto_y)
            valor_dy = evaluar_funcion(derivadas['d/dy'], None, None, punto_x, punto_y)
            
            col5, col6, col7 = st.columns(3)
            
            with col5:
                st.metric("f({:.1f}, {:.1f})".format(punto_x, punto_y), f"{valor_func:.4f}")
            
            with col6:
                st.metric("∂f/∂x({:.1f}, {:.1f})".format(punto_x, punto_y), f"{valor_dx:.4f}")
            
            with col7:
                st.metric("∂f/∂y({:.1f}, {:.1f})".format(punto_x, punto_y), f"{valor_dy:.4f}")
            
            # Gradiente
            st.subheader("∇ Gradiente")
            st.write(f"**Gradiente en ({punto_x}, {punto_y}):**")
            st.latex(f"\\nabla f({punto_x}, {punto_y}) = ({valor_dx:.4f}, {valor_dy:.4f})")
            
            # Magnitud del gradiente
            magnitud = np.sqrt(valor_dx**2 + valor_dy**2)
            st.write(f"**Magnitud del gradiente:** {magnitud:.4f}")
            
            # Dirección del gradiente
            if magnitud > 0:
                angulo = np.degrees(np.arctan2(valor_dy, valor_dx))
                st.write(f"**Dirección del gradiente:** {angulo:.2f}°")
            
        except Exception as e:
            st.error(f"Error al evaluar en el punto: {str(e)}")
        
        # Opción para mostrar gráficos
        st.subheader("📈 Visualización")
        mostrar_grafico = st.checkbox("Mostrar gráficos 3D", value=True)
        
        if mostrar_grafico:
            with st.spinner("Generando gráficos..."):
                fig = crear_grafico_3d(
                    func, 
                    (x_min, x_max), 
                    (y_min, y_max), 
                    derivadas, 
                    (punto_x, punto_y)
                )
                
                if fig:
                    st.pyplot(fig)
        
        # Información adicional
        with st.expander("ℹ️ Información sobre Derivadas Parciales"):
            st.markdown("""
            **¿Qué son las derivadas parciales?**
            
            Las derivadas parciales miden la tasa de cambio de una función multivariable con respecto a una de sus variables, manteniendo las otras constantes.
            
            **Notación:**
            - ∂f/∂x: Derivada parcial de f con respecto a x
            - ∂f/∂y: Derivada parcial de f con respecto a y
            
            **Gradiente:**
            El gradiente ∇f es un vector que contiene todas las derivadas parciales:
            ∇f = (∂f/∂x, ∂f/∂y)
            
            **Interpretación geométrica:**
            - El gradiente apunta en la dirección de mayor crecimiento de la función
            - La magnitud del gradiente indica qué tan rápido crece la función en esa dirección
            - Las derivadas parciales representan las pendientes de las curvas obtenidas al cortar la superficie con planos paralelos a los ejes
            
            **Sintaxis para funciones:**
            - Potencias: x**2, y**3
            - Trigonométricas: sin(x), cos(y), tan(x*y)
            - Exponencial: exp(x), exp(x*y)
            - Logarítmica: log(x), log(x**2 + y**2)
            - Operaciones: +, -, *, /, **
            """)
    
    else:
        st.error(f"Error en la función ingresada: {derivadas}")
        st.info("Verifica la sintaxis de tu función. Ejemplos válidos: x**2 + y**2, sin(x)*cos(y), exp(x*y)")

# Pie de página
st.markdown("---")
st.markdown("🔧 Desarrollado con Streamlit | ∂ Calculadora de Derivadas Parciales")