import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
import io
import xlsxwriter
import numpy as np
# PostgreSQL connection
DATABASE_URL = "postgresql://postgres:Lolly%40sql@localhost:5432/well log data"
engine = create_engine(DATABASE_URL)

# Function to load data from the database
def load_data():
    query = "SELECT * FROM well_log_data"
    return pd.read_sql(query, engine)

# Function to load sample data
def load_sample_data():
    df = pd.read_csv("log.csv")
    # df.drop(["DPOR"])
    return df

# Initialize session state for df if not already present
if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()  # Empty DataFrame as default


# Function to plot well log data
def plot_well_logs(df, columns):
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b' ,'g', 'r', 'c', 'm', 'y']  # Define a list of colors
    num_cols = len(columns)
    
    fig, axes = plt.subplots(1, num_cols, figsize=(5 * num_cols, 20), sharey=True)

    if num_cols == 1:
        axes = [axes]  # Ensure axes is iterable if only one column is selected

    for ax, column, color in zip(axes, columns, colors[:num_cols]):
        
        ax.plot(df[column], df['Depth'], label=column, color=color)
        ax.invert_yaxis()
        ax.set_xlabel(column)
        ax.set_ylabel("Depth")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plot_image_path = "well_log_plots.png"
    plt.savefig(plot_image_path)

    st.pyplot(fig)

def plot_cross_plots(df, neutron_column, density_column):
    # Create a figure and the axis for the plot
    fig, ax = plt.subplots(figsize=(4, 8))

    # Plot the neutron porosity (shared scale for both logs)
    ax.plot(df[neutron_column], df['Depth'], color='blue', label=f"{neutron_column} (Neutron Porosity)", linewidth = 0.5)

    # Plot the density porosity on the same axis
    ax.plot(df[density_column], df['Depth'], color='green', label=f"{density_column} (Density Porosity)",linewidth = 0.5)

    # Set labels and title
    ax.set_xlabel("Porosity (%)")  # Shared x-axis for porosity
    ax.set_ylabel('Depth')
    ax.set_title(f"{neutron_column} and {density_column} vs Depth")

    # Invert the vertical axis (depth increases downward)
    ax.invert_yaxis()
    ax.grid(True)
    # Show legend
    ax.legend()

    # Adjust layout and display the plot
    plt.tight_layout()
    st.pyplot(fig)



# Function to calculate porosity
def calculate_porosity(df, rho_m, rho_f):
    if 'RHOB' not in df.columns:
        st.error("Density (RHOB) column not found in data.")
        return df
    df['porosity'] = (rho_m - df['RHOB']) / (rho_m - rho_f)
    st.session_state.df = df  # Update session state
    return df
def classify_lithology(row):
    GR = row['GR']
    DPOR = row['DPOR']
    CNLS = row['CNLS']
    
    # Basic lithology classification based on GR, DPOR, and CNLS
    vsh = row['Vsh']
    if vsh >= 0.5:
        return 'Shale'
    

    else:
        if GR > 75:
            return 'Shale'
        elif 50 < GR <= 75 and DPOR > 10 and CNLS > 25:  # Medium GR, high porosity and CNLS suggest sandstone
            return 'Sandstone'
        elif GR <= 50 and DPOR < 10 and CNLS < 25:  # Low GR, low porosity suggest limestone
            return 'Limestone'
        else:
            return 'Unknown'
def calculate_shale_volume(df):
    # Determine GR_min and GR_max from the dataset
    GR_min = df['GR'].min()  # Clean sand value (lowest GR)
    GR_max = df['GR'].max()  # Shale value (highest GR)

    # Calculate shale volume
    df['Vsh'] = (df['GR'] - GR_min) / (GR_max - GR_min)
    df['Vsh'] = df['Vsh'].clip(lower=0, upper=1)  # Ensure Vsh is between 0 and 1
    st.session_state.df = df  # Update session state
    return df, GR_min, GR_max
# Function to classify lithology for the entire dataset
def apply_lithology_classification(df):
    df, GR_min, GR_max = calculate_shale_volume(df)
    df['Lithology'] = df.apply(classify_lithology, axis=1)
    lithology_colors = {'Shale': 'brown', 'Sandstone': 'yellow', 'Limestone': 'blue', 'Unknown': 'gray'}
    df['Lithology Color'] = df['Lithology'].map(lithology_colors)  # Add a color column for plotting
    st.session_state.df = df  # Update session state
    return df, GR_min, GR_max


def plot_gamma_with_lithology(df):
    plt.figure(figsize=(10, 15))

    # Define colors for lithology
    lithology_colors = {
        'Shale': 'brown',
        'Sandstone': 'yellow',
        'Limestone': 'blue',
        'Unknown': 'gray'
    }
    
    # Plot the gamma log
    plt.plot(df['GR'], df['Depth'], label='Gamma Ray', color='black')

    # Shade areas based on lithology
    for lithology, color in lithology_colors.items():
        lith_df = df[df['Lithology'] == lithology]
        plt.fill_betweenx(lith_df['Depth'], lith_df['GR'], color=color, alpha=0.5, label=lithology)

    plt.gca().invert_yaxis()  # Invert y-axis for depth
    plt.xlabel('Gamma Ray (API)')
    plt.ylabel('Depth (m)')
    plt.title('Gamma Ray Log with Lithology Shading')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_image_path = "log_plots.png"
    plt.savefig(plot_image_path)
    st.pyplot(plt)
# Function to calculate water saturation
def calculate_saturation(df, a, m, n, rw):
    if 'porosity' not in df.columns:
        st.error("Porosity not calculated yet. Please calculate porosity first.")
        return df
    if 'RILD' not in df.columns:
        st.error("Resistivity (RILD) column not found in data.")
        return df

    # Archie's equation for saturation calculation
    df['water_saturation'] = ((a * rw) / (df['porosity']**m * df['RILD'])) ** (1/n)
    st.session_state.df = df  # Update session state
    return df

# App Title
st.title("Well Log Insights")

# Data Option: Use sample data or upload data
st.markdown("Don't have data to upload?     \nClick the button below to try sample data.")
use_sample_data = st.button("Use Sample Data")

if use_sample_data:
    st.session_state.df = load_sample_data()
    st.success("Using sample data.")
else:
    uploaded_file = st.file_uploader("Upload Well Log Data (CSV format)", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = pd.read_csv(uploaded_file)
        st.session_state.df.to_sql('well_log_data', engine, if_exists='replace', index=False)
        st.success("Data uploaded successfully and stored in PostgreSQL.")
    else:
        st.warning("No data uploaded. Please upload a file or use sample data.")

df = st.session_state.df  # Load df from session state

# Display data if it exists
if not st.session_state.df.empty:
    st.subheader("Well Log Data")
    st.dataframe(df)

    # Select columns to plot
    st.subheader("Plot logs with depth or crossplot")
    plot_type = st.selectbox("Select Plot Type", ["Normal Plot", "Cross Plot"])
    

    # Plot well log data
    
    if plot_type == "Normal Plot":
            selected_logs = st.multiselect("", df.columns[1:], default=df.columns[1:4])
            plot_well_logs(st.session_state.df, selected_logs)
    else:
        columns_without_depth = [col for col in df.columns if col != 'Depth']
        default_x_column = 'DPOR' if 'DPOR' in columns_without_depth else columns_without_depth[0]
        default_y_column = 'CNLS' if 'CNLS' in columns_without_depth else columns_without_depth[0]
       
         # Streamlit selectbox with default preselected columns
        y_column = st.selectbox("Select X-axis for Cross Plot", columns_without_depth, index=columns_without_depth.index(default_x_column))
        x_column = st.selectbox("Select Y-axis for Cross Plot", columns_without_depth, index=columns_without_depth.index(default_y_column))
        
# Call the cross plot function only when both columns are selected
        if x_column and y_column:
    
            plot_cross_plots(st.session_state.df , x_column, y_column)
            
    st.subheader("Lithology Classification")


    if st.button("Classify Lithology"):
        st.session_state.df, GR_min, GR_max = apply_lithology_classification(st.session_state.df)
        st.success("Lithology classified successfully.")
        
        st.write(f"GR_min (Clean Sand): {GR_min:.2f} API")
        st.write(f"GR_max (100% Shale): {GR_max:.2f} API")
        
        # Plot gamma log with lithology shading
        plot_gamma_with_lithology(st.session_state.df)


    # Porosity calculation form
    with st.form(key='porosity_form'):
        st.subheader("Calculate Porosity")
        rho_m = st.number_input("Matrix Density (g/cm³)", value=2.65, step=0.01)
        rho_f = st.number_input("Fluid Density (g/cm³)", value=1.0, step=0.01)
        calculate_porosity_button = st.form_submit_button("Calculate Porosity")

        if calculate_porosity_button:
            st.session_state.df = calculate_porosity(st.session_state.df, rho_m, rho_f)
            st.success("Porosity calculated successfully.")
            st.dataframe(st.session_state.df[['Depth', 'porosity']])
    df = calculate_porosity(df,rho_m , rho_f)
    # Saturation calculation form
    with st.form(key='saturation_form'):
        st.subheader("Calculate Saturation")
        a = st.number_input("Formation Resistivity Factor (a)", value=1.0, step=0.01)
        m = st.number_input("Cementation Exponent (m)", value=2.0, step=0.01)
        n = st.number_input("Saturation Exponent (n)", value=2.0, step=0.01)
        rw = st.number_input("Water Resistivity (rw)", value=0.1, step=0.01)
        calculate_saturation_button = st.form_submit_button("Calculate Saturation")

        if calculate_saturation_button:
            st.session_state.df = calculate_saturation(st.session_state.df, a, m, n, rw)
            st.success("Saturation calculated successfully.")
            
            # Ensure the column exists before displaying it
            if 'water_saturation' in st.session_state.df.columns:
                st.dataframe(st.session_state.df[['Depth', 'water_saturation']])
            else:
                st.error("Error: Water saturation was not calculated properly.")

    df = calculate_saturation(df,a,m,n,rw)    
    # Hydrocarbon zone identification
    def identify_hydrocarbon_zones(df):
        if 'porosity' not in df.columns:
            st.error("Porosity not calculated yet.")
            return df

        oil_mask = (df['RILD'] > 50) & (df['RILD'] <= 100) & (df['porosity'] > 0.1)
        gas_mask = (df['RILD'] > 100) & (df['porosity'] > 0.15) & (df['DPOR'] - df['CNLS'] > 10)
       

       
        df.loc[gas_mask, 'hydrocarbon_zone'] = 'Gas'
        df.loc[oil_mask, 'hydrocarbon_zone'] = 'Oil'

        st.session_state.df = df  # Update session state
        return df
    st.subheader("Plot hydrocarbon zone with depth")
    if st.button("Identify Hydrocarbon Zones"):
        if 'porosity' not in st.session_state.df.columns:
            st.warning("Porosity needs to be calculated before identifying hydrocarbon zones.")
        else:
            st.session_state.df = identify_hydrocarbon_zones(st.session_state.df)
            st.success("Hydrocarbon zones identified successfully.")
           

            # Plot hydrocarbon zones
            fig, ax = plt.subplots(figsize=(5, 10))
            ax.fill_betweenx(df['Depth'], 0, 1, where=(df['hydrocarbon_zone'] == 'Gas'), color='yellow', label='Gas')
            ax.fill_betweenx(df['Depth'], 0, 1, where=(df['hydrocarbon_zone'] == 'Oil'), color='green', label='Oil')
            # ax.fill_betweenx(df['Depth'], 0, 1, where=(df['hydrocarbon_zone'] == 'Water'), color='blue', label='Water')
            ax.invert_yaxis()
            ax.set_xlabel("Hydrocarbon Zones")
            ax.set_ylabel("Depth")
            ax.legend()
            plot_image_path = "hydrocarbon_zone_plots.png"
            plt.savefig(plot_image_path)
            st.pyplot(fig)
    df = identify_hydrocarbon_zones(df)
    # Data Export and Reporting
    st.subheader('Data Export and Reporting')

    if st.button('Download Well Log Report'):
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            st.session_state.df.to_excel(writer, sheet_name='Well Log Data', index=False)
            
            # Add the well log plot image to the Excel file
            workbook = writer.book
            worksheet = workbook.add_worksheet('Plots')
            worksheet.insert_image('A1', 'well_log_plots.png')
            worksheet.insert_image('A100', 'log_plots.png')
            worksheet.insert_image('P100', 'hydrocarbon_zone_plots.png')

           
        st.download_button(
            label='Download Excel File with plots',
            data=buffer.getvalue(),
            file_name='well_log_data_with_plots.xlsx',
            mime='application/vnd.ms-excel'
        )
