import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import matplotlib.animation as animation
import pandas as pd
import io

# Função para a calculadora de temperatura (seu código original)
def l_shaped_temp_variable_properties(t_final, len_x, len_y, len_w, len_z, delta,
                                        ks_data, cs_data, ds_data, hg_data,
                                        tamb, t_initial):

    # Dimensões da grade
    nx = int(len_x / delta) + 1
    ny = int(len_y / delta) + 1
    nw = int(len_w / delta) + 1
    nz = int(len_z / delta) + 1

    print(f"nx: {nx}, ny: {ny}, nw: {nw}, nz: {nz}")

    # Criar a geometria em forma de L
    domain = np.full((nx, ny), 1, dtype=np.float64)
    for i in range(nx):
        for j in range(ny):
            # 0 para fora do domínio, 1 para dentro, 2 para plano de simetria, 3 para superfície externa
            if (j == 0):
                domain[i, j] = 2  # Células do plano de simetria da alma são marcadas com 2
            if (i == 0):
                domain[i, j] = 3  # Células da borda externa do flange são marcadas com 3
            if (i * delta > len_z and j * delta > len_w):
                domain[i, j] = 0  # Células fora do domínio são marcadas com 0
            if (j * delta == len_y and i * delta < len_z):
                domain[i, j] = 3  # Células da borda direita no plano de simetria são marcadas com 2
            if (i * delta == len_x and j * delta < len_w):
                domain[i, j] = 2  # Células da borda superior na superfície externa são marcadas com 3
            if (i * delta == len_z and j * delta > len_w):
                domain[i, j] = 3  # Células da borda interna horizontal na superfície externa são marcadas com 3
            if (j * delta == len_w and i * delta > len_z):
                domain[i, j] = 3  # Células da borda interna vertical na superfície interna são marcadas com 3

    # Funções de interpolação
    ks_interp = interp1d(ks_data[0], ks_data[1], kind='linear', fill_value="extrapolate")
    cs_interp = interp1d(cs_data[0], cs_data[1], kind='linear', fill_value="extrapolate")
    ds_interp = interp1d(ds_data[0], ds_data[1], kind='linear', fill_value="extrapolate")
    hg_interp = interp1d(hg_data[0], hg_data[1], kind='linear', fill_value="extrapolate")


    # Inicialização da matriz de temperatura
    t = np.zeros((nx, ny), dtype=np.float64)
    if isinstance(t_initial, (int, float)):
        t[:] = t_initial
    else:
        t = np.copy(t_initial) # Assume que t_initial tem o formato correto

    t_prev = np.copy(t)

    # Pontos para monitorar a temperatura
    pontos = [(0, 0), (int(nz/2), int(ny/2)), (int(nx-1) , 0)]
    temperaturas = {p: [] for p in pontos}

    # Lista para armazenar os campos de temperatura para a animação
    temperature_fields = []  # Aqui está a lista para a animação


    # Loop no tempo
    for tempo in np.arange(0, t_final, 1): #reduzi o passo de tempo pq alpha é variavel
        #print(f"Tempo atual: {tempo}")  # Debug: Imprimir o tempo atual
        # Loop espacial
        for i in range(0, nx):
            for j in range(0, ny):
                if domain[i, j] != 0:  # Calcula apenas dentro do domínio
                    # Calcula propriedades dependentes da temperatura (USANDO A TEMPERATURA DO PASSO ANTERIOR)
                    ks = ks_interp(t_prev[i, j])
                    cs = cs_interp(t_prev[i, j])
                    ds = ds_interp(t_prev[i, j])
                    hg = hg_interp(t_prev[i, j])

                    # Cálculo da difusividade térmica
                    alpha = ks / (cs * ds)

                    # Definição do número de Biot (Ms)
                    ms = hg * (delta / 1000) / ks

                    # Definição do número de Fourier (Ml)
                    ml = max(4, ms + 3, 2 * ms + 2) #PODE SER QUE PRECISE AUMENTAR ISSO AQUI

                    # Cálculo do passo de tempo (agora dentro do loop, pois depende da temperatura)
                    dt = (delta / 1000)**2 / (ml * alpha)
                    if (dt > 1): dt = 1  #Limitando dt para no máximo 1 segundo

                    # Condição interna
                    if i > 0 and i < nx - 1 and j > 0 and j < ny - 1 and domain[i,j] == 1:
                        t[i, j] = (1 / ml) * (t_prev[i + 1, j] + t_prev[i - 1, j] + t_prev[i, j + 1] + t_prev[i, j - 1]) + t_prev[i, j] * (ml - 4) / ml
                    # Condições de contorno (convecção)
                    else:
                        # Superfície externa Flange
                        if i == 0 and j < ny - 1 and domain[i,j] == 3:
                            t[i, j] = (ms / ml) * tamb + (1 / ml) * (t_prev[i, j - 1] + t_prev[i, j + 1] + t_prev[i + 1, j]) + t_prev[i, j] * (1 - (ms + 3) / ml)
                        # Plano de simetria Vertical
                        if i == nx - 1 and domain[i, j] == 2:
                            t[i, j] = (1 / ml) * (t_prev[i, j - 1] + t_prev[i, j + 1] + t_prev[i - 1, j]) + t_prev[i, j] * (ml - 3) / ml
                        # Plano de Simetria Horizontal
                        if j == 0 and i < nx - 1 and domain[i, j] == 2:
                            t[i, j] = (1 / ml) * (t_prev[i - 1, j] + t_prev[i + 1, j] + t_prev[i, j + 1]) + t_prev[i, j] * (ml - 3) / ml
                        # Superfície superior Flange
                        if j == ny - 1 and domain[i, j] == 3:
                            t[i, j] = (ms / ml) * tamb + (1 / ml) * (t_prev[i - 1, j] + t_prev[i + 1, j] + t_prev[i, j - 1]) + t_prev[i, j] * (1 - (ms + 3) / ml)
                        # Canto meio superfície Flange
                        if i == 0 and j == 0 and domain[i, j] == 3:
                            t[i, j] = (ms / ml) * tamb + (1 / ml) * (t_prev[i, j + 1] + t_prev[i + 1, j]) + t_prev[i, j] * (1 - ( ms + 2) / ml)
                        # Canto superior Flange esquerdo
                        if i == 0 and j == ny - 1 and domain[i, j] == 3:
                            t[i, j] = 2 * (ms / ml) * tamb + (1 / ml) * (t_prev[i, j - 1] + t_prev[i + 1, j]) + t_prev[i, j] * (1 - (2 * ms + 2) / ml)
                         # Canto meio plano de simetria
                        if i == nx - 1 and j == 0 and domain[i, j] == 2:
                            t[i, j] = (1 / ml) * (t_prev[i, j + 1] + t_prev[i - 1, j]) + t_prev[i, j] * (ml - 2) / ml
                        # Canto plano de simetria superfície interna
                        if i == nx - 1 and j == int(nw-1) and domain[i, j] == 3:
                            t[i, j] = (ms / ml) * tamb + (1 / ml) * (t_prev[i, j - 1] + t_prev[i - 1, j]) + t_prev[i, j] * (1 - ( ms + 2) / ml)
                        # Canto superficie flange interno
                        if i == int(nz-1) and j == ny-1 and domain[i, j] == 3:
                            t[i, j] = 2 * (ms / ml) * tamb + (1 / ml) * (t_prev[i, j - 1] + t_prev[i - 1, j]) + t_prev[i, j] * (1 - (2 * ms + 2) / ml)
                        #Condições de contorno nas bordas internas do L
                        # Borda interna vertical
                        if i == int(nz-1) and j >= int(nw-1) and j < ny - 1 and domain[i, j] == 3:
                            t[i, j] = (ms / ml) * tamb + (1 / ml) * (t_prev[i, j - 1] + t_prev[i, j + 1] + t_prev[i - 1, j]) + t_prev[i, j] * (1 - (ms + 3) / ml)

                        # Borda interna horizontal
                        if j == int(nw-1) and i >= int(nz-1) and i < nx - 1 and domain[i, j] == 3:
                            t[i, j] = (ms / ml) * tamb + (1 / ml) * (t_prev[i - 1, j] + t_prev[i + 1, j] + t_prev[i, j - 1]) + t_prev[i, j] * (1 - (ms + 3) / ml)


        # Armazenar temperaturas nos pontos monitorados
        for p in pontos:
            temperaturas[p].append(t[p])

        # Armazenar o campo de temperatura para a animação
        temperature_fields.append(np.copy(t)) # Salva uma cópia do campo de temperatura

        t_prev = np.copy(t)

        # Debug: Imprimir a matriz de temperatura em alguns pontos do tempo
        #if tempo % (t_final / 10) == 0:
            #print(f"Temperatura no tempo {tempo}:")
            #print(t)

    return temperature_fields, temperaturas, 1, pontos, nx, ny, nz #dt = 1 para facilitar plotagem no exemplo de uso


# Funções auxiliares
def calculate_cooling_rate(temperatures, times, start_temp, end_temp):
    """Calcula a taxa de resfriamento entre duas temperaturas."""
    try:
        start_index = next(i for i, temp in enumerate(temperatures) if temp <= start_temp)
        end_index = next(i for i, temp in enumerate(temperatures[start_index:]) if temp <= end_temp) + start_index
        time_diff = times[end_index] - times[start_index]
        temp_diff = start_temp - end_temp
        cooling_rate = temp_diff / time_diff if time_diff > 0 else float('inf')
        return cooling_rate
    except StopIteration:
        return float('inf')  # Resfriamento completo

def save_temperature_data(tempos, temperaturas, pontos):
    """Salva os dados de temperatura em um arquivo TXT."""
    output = io.StringIO()
    output.write("Time (s)")
    for p in pontos:
        output.write(f", Point {p}")
    output.write("\n")

    # Determine o comprimento mínimo das listas de temperatura
    min_len = min(len(temperaturas[p]) for p in pontos)
    min_len = min(min_len, len(tempos)) #garantindo que não vai extrapolar os tempos

    for i in range(min_len):
        output.write(f"{tempos[i]:.1f}")
        for p in pontos:
            output.write(f", {temperaturas[p][i]:.2f}")
        output.write("\n")

    return output.getvalue()

# Configuração da página Streamlit
st.set_page_config(
    page_title="H-Beam Heat Transfer Calculator",
    #page_icon=":desktop_computer:",
    page_icon=":technologist:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Variáveis de estado da sessão
if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = None
if 'username' not in st.session_state:
    st.session_state['username'] = None

# Página de Login
def login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Entrar"):
        # Simulação de autenticação (substitua por um sistema real)
        if username == "roberto.tiburcio" and password == "canito":
            st.session_state['authentication_status'] = True
            st.session_state['username'] = username
            st.success("Login sucessful! Please select calculator from the menu.")
        elif username == "vinicius.ottani" and password == "cardoso":
            st.session_state['authentication_status'] = True
            st.session_state['username'] = username
            st.success("Login sucessful! Please select calculator from the menu.")
        elif username == "marcelo.rebellato" and password == "arantes":
            st.session_state['authentication_status'] = True
            st.session_state['username'] = username
            st.success("Login sucessful! Please select calculator from the menu.")
        elif username == "antonio.gorni" and password == "augusto":
            st.session_state['authentication_status'] = True
            st.session_state['username'] = username
            st.success("Login sucessful! Please select calculator from the menu.")
        elif username == "jose.bacalhau" and password == "britti":
            st.session_state['authentication_status'] = True
            st.session_state['username'] = username
            st.success("Login sucessful! Please select calculator from the menu.")
        else:
            st.session_state['authentication_status'] = False
            st.error("Login failed. Please try again.")

# Página da Calculadora
def calculator():
    st.title("H-Beam Heat Transfer Calculator")

    # Opção de entrada de dados
    data_option = st.radio("Select data input type:", ("Typing", "Upload CSV File"))

    # Inicializar dicionários para os dados
    input_data = {}

    if data_option == "Typing":
        st.subheader("Input Data (Typing)")
        # Dados de entrada numéricos
        input_data['t_final'] = st.number_input("Total Cooling Time (s)", value=400.0, key="t_final")
        input_data['len_x'] = st.number_input("Half Profile Height  (mm)", value=150.0, key="len_x")
        input_data['len_y'] = st.number_input("Half Flange Width (mm)", value=100.0, key="len_y")
        input_data['len_w'] = st.number_input("Half Web Thickness (mm)", value=20.0, key="len_w")
        input_data['len_z'] = st.number_input("Flange Thickness (mm)", value=30.0, key="len_z")
        input_data['delta'] = st.number_input("Mesh Size (mm)", value=5.0, key="delta")
        input_data['tamb'] = st.number_input("Environment Temperature (°C)", value=30.0, key="tamb")
        input_data['t_initial'] = st.number_input("Initial Temperature (°C)", value=850.0, key="t_initial")

        # Dados de propriedades variáveis (listas de números)
        st.write("Thermal Conductivity (ks_data):")
        input_data['ks_temperatures'] = [float(x) for x in st.text_input("Temperatures (°C), separated by comma", value="20, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1075", key="ks_temperatures").split(",")]
        input_data['ks_values'] = [float(x) for x in st.text_input("Values (W/m.K), separated by comma", value="37, 36.5, 34.5, 32.6, 30.7, 29.0, 25.8, 26.2, 26.8, 27.4, 28.1, 28.4", key="ks_values").split(",")]

        st.write("Specific Heat (cs_data):")
        input_data['cs_temperatures'] = [float(x) for x in st.text_input("Temperatures (°C), separated by comma", value="20, 600, 650, 700, 750, 775, 800, 850, 900, 950, 1000, 1050, 1075", key="cs_temperatures").split(",")]
        input_data['cs_values'] = [float(x) for x in st.text_input( "Values (J/kg.K), separated by comma", value="770, 765, 824, 892, 971, 1014, 645, 647, 648, 650, 651, 653, 655", key="cs_values").split(",")]

        st.write("Density (ds_data):")
        input_data['ds_temperatures'] = [float(x) for x in st.text_input("Temperatures (°C), separated by comma", value="20, 600, 1300", key="ds_temperatures").split(",")]
        input_data['ds_values'] = [float(x) for x in st.text_input( "Values (kg/m3), separated by comma", value="7850, 7800, 7332", key="ds_values").split(",")]

        st.write("Global Heat Transfer Coeffincient (hg_data):")
        input_data['hg_temperatures'] = [float(x) for x in st.text_input("Temperatures (°C), separated by comma", value="20, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200", key="hg_temperatures").split(",")]
        input_data['hg_values'] = [float(x) for x in st.text_input( "Values (W/m^2.K), separated by comma", value="15, 64, 72, 80, 89, 98, 108, 120, 132, 145, 158, 173", key="hg_values").split(",")]

        st.write("Temperatures for calculating the cooling rate (°C):")
        start_temp = st.number_input("Initial temperature for calculating the cooling rate (°C)", value=800.0)
        end_temp = st.number_input("Final temperature for calculating the cooling rate (°C)", value=700.0)
    
    else:  # Upload de Arquivo CSV
        st.subheader("Input Data (Upload CSV File)")
        uploaded_file = st.file_uploader("Load CSV File", type=["csv"])

        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)

                # Extrair dados do DataFrame
                input_data['t_final'] = float(df['t_final'][0])
                input_data['len_x'] = float(df['len_x'][0])
                input_data['len_y'] = float(df['len_y'][0])
                input_data['len_w'] = float(df['len_w'][0])
                input_data['len_z'] = float(df['len_z'][0])
                input_data['delta'] = float(df['delta'][0])
                input_data['tamb'] = float(df['tamb'][0])
                input_data['t_initial'] = float(df['t_initial'][0])

                input_data['ks_temperatures'] = df['ks_temperatures'].dropna().tolist()
                input_data['ks_values'] = df['ks_values'].dropna().tolist()
                input_data['cs_temperatures'] = df['cs_temperatures'].dropna().tolist()
                input_data['cs_values'] = df['cs_values'].dropna().tolist()
                input_data['ds_temperatures'] = df['ds_temperatures'].dropna().tolist()
                input_data['ds_values'] = df['ds_values'].dropna().tolist()
                input_data['hg_temperatures'] = df['hg_temperatures'].dropna().tolist()
                input_data['hg_values'] = df['hg_values'].dropna().tolist()

                st.success("CSV data loaded successfully!")

            except Exception as e:
                st.error(f"Error loading CSV file: {e}")
                return  # Abortar se houver erro no upload

    # Botão de Calcular
    if st.button("Calculate"):
        try:
            # Preparar os dados para a função de cálculo
            t_final = input_data['t_final']
            len_x = input_data['len_x']
            len_y = input_data['len_y']
            len_w = input_data['len_w']
            len_z = input_data['len_z']
            delta = input_data['delta']
            tamb = input_data['tamb']
            t_initial = input_data['t_initial']

            ks_data = (input_data['ks_temperatures'], input_data['ks_values'])
            cs_data = (input_data['cs_temperatures'], input_data['cs_values'])
            ds_data = (input_data['ds_temperatures'], input_data['ds_values'])
            hg_data = (input_data['hg_temperatures'], input_data['hg_values'])

            # Executar a simulação
            temperature_fields, temperaturas, dt, pontos, nx, ny, nz = l_shaped_temp_variable_properties(
                t_final, len_x, len_y, len_w, len_z, delta,
                ks_data, cs_data, ds_data, hg_data,
                tamb, t_initial
            )

            # Definir os limites da escala de temperatura
            vmin = 600
                #min(input_data['ks_temperatures']), min(input_data['cs_temperatures']),min(input_data['ds_temperatures']),min(input_data['hg_temperatures']))  # Temperatura mínima
            vmax = t_initial  # Temperatura máxima

            # Intervalo de tempo para salvar a animação (em segundos)
            animation_interval = 10

            # Criar uma nova lista com os campos de temperatura a cada 'animation_interval' segundos
            sampled_temperature_fields = temperature_fields[::animation_interval]

            # Criar a animação
            fig, ax = plt.subplots()
            im = ax.imshow(sampled_temperature_fields[0].T, cmap='hot', origin='lower', extent=[0, len_x, 0, len_y], animated=True, vmin=vmin, vmax=vmax) # vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(im, label='Temperature (°C)')
            ax.set_xlabel(' 1/2 Profile Height (mm)')
            ax.set_ylabel('1/2 Flange Width (mm)')
            ax.set_title('Temperature Distribution 1/4 H-Beam Section')

            # Adicionar os pontos e rótulos à animação
            point_colors = ['black', 'black', 'black']  # Cores para cada ponto
            point_labels_animation = ['Flange Surface', 'Flange Middle', 'Web']  # Rótulos para os pontos

            for i, p in enumerate(pontos):
                x_pixel = p[0] * delta
                y_pixel = p[1] * delta
                ax.plot(x_pixel, y_pixel, marker='o', color=point_colors[i], markersize=5)  # Ajuste o tamanho dos marcadores conforme necessário
                # Adicionar o rótulo ao lado dos 3 pontos
                ax.text(x_pixel + 2, y_pixel + 1, point_labels_animation[i], color='black', fontsize=8)  # Ajuste a posição e o tamanho da fonte conforme necessário
            

            time_text = ax.text(0.75, 0.90, '', transform=ax.transAxes, color='black') #Posiciona o texto do tempo

            def update_fig(num):
                im.set_array(sampled_temperature_fields[num].T)
                time_text.set_text(f'Time: {num*animation_interval*dt:.1f} s') #Atualiza o texto do tempo
                return [im, time_text]

            ani = animation.FuncAnimation(fig, update_fig, frames=len(sampled_temperature_fields), blit=True)

            # Salvar a animação como um arquivo GIF
            try:
                ani.save('temperature_animation.gif', writer='pillow') #Certifique-se de ter o Pillow instalado (`pip install Pillow`)
                print("Animation saved as temperature_animation.gif")
            except Exception as e:
                print(f"Error saving animation (Pillow might not be installed): {e}")


            plt.close(fig)  # Importante: Feche a figura para liberar memória

            # Exibir a animação no Streamlit
            st.subheader("Temperature Distribution Animation")
            try:
                st.image("temperature_animation.gif", use_container_width=True)
            except FileNotFoundError:
                st.error("Animation file not found. Please check if the animation was saved correctly.")
            except Exception as e:
                st.error(f"Error displaying animation: {e}")

            # Plotar a evolução da temperatura no tempo para os pontos monitorados
            st.subheader("Temperature Evolution at Monitored Points")
            tempos = np.arange(0, t_final, dt)
            fig_temp, ax_temp = plt.subplots()  # Criar uma nova figura para o gráfico de temperatura
            
            # Novos rótulos para os pontos
            # Access nx, ny, nz from the function's return
            point_labels = {
                (0, 0): 'Flange Surface',
                (int(nz/2), int(ny/2)): 'Flange Middle',
                (int(nx-1) , 0): 'Web Middle'
            }

            for p in temperaturas:
                ax_temp.plot(tempos, temperaturas[p], label=point_labels[p])

            ax_temp.set_xlabel('Time (seconds)')
            ax_temp.set_ylabel('Temperature (°C)')
            ax_temp.set_title('Temperature Evolution at Monitored Points')
            ax_temp.legend()
            st.pyplot(fig_temp)  # Exibir o gráfico no Streamlit

            # Armazenar os resultados no session state
            st.session_state['tempos'] = tempos
            st.session_state['temperaturas'] = temperaturas
            st.session_state['pontos'] = pontos
            st.session_state['start_temp'] = start_temp
            st.session_state['end_temp'] = end_temp
            st.session_state['nx'] = nx  # Store nx in session state
            st.session_state['ny'] = ny  # Store ny in session state
            st.session_state['nz'] = nz  # Store nz in session state

        except Exception as e:
            st.error(f"Error during calculation: {e}")

    # Opção para salvar os dados do gráfico em um arquivo TXT
    st.subheader("Save Temperature Evolution Data")
    if 'tempos' in st.session_state and 'temperaturas' in st.session_state and 'pontos' in st.session_state:
        if st.button("Generate TXT File"):
            tempos = st.session_state['tempos']
            temperaturas = st.session_state['temperaturas']
            pontos = st.session_state['pontos']
            txt_data = save_temperature_data(tempos, temperaturas, pontos)
            st.download_button(
                label="Download TXT file",
                data=txt_data,
                file_name="temperature_evolution.txt",
                mime="text/plain",
            )

    # Cálculo da taxa de resfriamento
    st.subheader("Cooling Rate Calculation")
    if 'tempos' in st.session_state and 'temperaturas' in st.session_state and 'pontos' in st.session_state and 'nx' in st.session_state and 'ny' in st.session_state and 'nz' in st.session_state:
        tempos = st.session_state['tempos']
        temperaturas = st.session_state['temperaturas']
        pontos = st.session_state['pontos']
        start_temp = st.session_state['start_temp']
        end_temp = st.session_state['end_temp']
        nx = st.session_state['nx']  # Retrieve nx from session state
        ny = st.session_state['ny']  # Retrieve ny from session state
        nz = st.session_state['nz']  # Retrieve nz from session state

        # Rótulos para os pontos
        point_labels_cooling = {
            (0, 0): 'Flange Surface',
            (int(nz/2), int(ny/2)): 'Flange Middle',
            (int(nx-1), 0): 'Web Middle'
        }

        cooling_rates = {}
        for p in pontos:
            cooling_rate = calculate_cooling_rate(temperaturas[p], tempos, start_temp, end_temp)
            cooling_rates[p] = cooling_rate

        st.write("Cooling rate (°C/s):")
        for p, rate in cooling_rates.items():
            st.write(f"{point_labels_cooling[p]}: {rate:.2f} °C/s")

# Sidebar para navegação
menu = ["Login", "Calculator"]
choice = st.sidebar.selectbox("Menu", menu)

# Lógica de roteamento
if st.session_state['authentication_status'] != True:
    login()
else:
    if choice == "Calculator":
        calculator()
    elif choice == "Login":
        st.write(f"Wellcome, {st.session_state['username']}!")
        if st.button("Logout"):
            st.session_state['authentication_status'] = False
            st.session_state['username'] = None
