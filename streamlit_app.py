import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except Exception:  # plotly not installed
    PLOTLY_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="Solar Pile Optimization Analysis",
    page_icon="☀️",
    layout="wide"
)
version="2.3"
# Title and description
st.title(f"☀️ Solar Pile Optimization Analysis V {version}")
st.markdown("""
This application analyzes solar array pile data to determine the optimal racking line 
and calculate required pile lengths. It compares three optimization methods and applies 
North-South constraints between adjacent rows.
""")

# Sidebar for configuration
st.sidebar.header("⚙️ Configuration")

# File upload
uploaded_file = st.sidebar.file_uploader(
    "Upload Excel File (Grading.xlsx)", 
    type=['xlsx', 'xls'],
    help="Upload your Excel file containing Array Pile Data"
)

sheet_name = st.sidebar.text_input("Sheet Name", value="Array Pile Data")
header_row_index = st.sidebar.number_input(
    "Header Row Index (0-based)", value=1, min_value=0,
    help="Adjust if your column headers are actually on a different row. If unsure, try 0."
)

# Configuration parameters
st.sidebar.subheader("Optimization Parameters")
apply_ns_constraints = st.sidebar.checkbox("Apply North-South Constraints", value=True)

# N-S constraint tuning (used only when Apply North-South Constraints is enabled)
st.sidebar.subheader("North-South Constraint Parameters")
ns_northing_diff_min_ft = st.sidebar.number_input(
    "N-S constraint: Northing diff minimum (ft)",
    value=20.0,
    min_value=0.0,
    help="If the northing difference between adjacent rows is below this, the close-spacing constraint logic is applied. Default = 20 ft."
)
ns_northing_diff_max_ft = st.sidebar.number_input(
    "N-S constraint: Northing diff maximum (ft)",
    value=60.0,
    min_value=0.0,
    help="If the northing difference is between the minimum and this maximum, the mid-spacing constraint logic is applied. Default = 60 ft."
)
ns_top_of_pile_tolerance_ft = st.sidebar.number_input(
    "N-S constraint: Top-of-pile vertical tolerance (ft)",
    value=2.0,
    min_value=0.0,
    help="Vertical tolerance used when comparing/adjusting adjacent-row top-of-pile elevations in the mid-spacing constraint logic. Default = 2.0 ft."
)
ns_top_of_pile_tolerance_close_ft = st.sidebar.number_input(
    "N-S constraint: Close Spaced Top-of-pile vertical tolerance (ft)",
    value=0.1,
    min_value=0.0,
    help="Vertical tolerance used when adjacent rows are closer than the minimum northing difference (close-spacing constraint). Default = 0.1 ft."
)

# Guardrail: ensure max >= min
if ns_northing_diff_max_ft < ns_northing_diff_min_ft:
    ns_northing_diff_max_ft = ns_northing_diff_min_ft
grading_width_ft = st.sidebar.number_input("Grading Width (ft)", value=1.0, min_value=0.1)
min_reveal_height = st.sidebar.number_input("Min Reveal Height (ft)", value=4.0, min_value=0.0)
max_reveal_height = st.sidebar.number_input("Max Reveal Height (ft)", value=5.0, min_value=0.0)

st.sidebar.subheader("Cost Weights")
cost_weight_grading = st.sidebar.number_input("Grading Weight ($/ft³)", value=10, min_value=0)
cost_weight_pile = st.sidebar.number_input("Pile Weight ($/ft)", value=1, min_value=0)

test_mode = st.sidebar.checkbox("Test Mode (Visualizations)", value=False)

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def load_data_from_excel(file, sheet_name='Array Pile Data'):
    """Loads ground survey data from Excel file."""
    try:
        df = pd.read_excel(file, sheet_name=sheet_name, header=1)
        required_columns = ['Easting', 'Northing', 'EG']
        if not all(col in df.columns for col in required_columns):
            st.error(f"Excel sheet must contain columns: {required_columns}")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading Excel file: {e}")
        return None

@st.cache_data
def read_array_pile_first_12(file_bytes, sheet='Array Pile Data', header_row=1):
    """Read the first 12 columns from Array Pile Data with adaptive header handling."""
    attempted_headers = [header_row]
    if header_row != 0:
        attempted_headers.append(0)  # Fallback attempt if custom header fails structure expectations

    last_error = None
    for hdr in attempted_headers:
        try:
            apd = pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet, header=hdr, usecols=range(12))

            # If we already have expected names, keep them; else rename Unnamed columns
            rename_map = {
                'Unnamed: 0': 'Point Number',
                'Unnamed: 1': 'InverterGroup Number',
                'Unnamed: 2': 'Tracker Motor Combiner Number',
                'Unnamed: 3': 'Row',
                'Unnamed: 4': 'Pile',
                'Unnamed: 5': 'INVGroup-ROW',
                'Unnamed: 6': 'Tracker Type',
                'Unnamed: 7': 'Pile Type(color)',
                'Unnamed: 8': 'Zone Area',
                'Unnamed: 9': 'Northing',
                'Unnamed: 10': 'Easting',
                'Unnamed: 11': 'EG',
            }
            apd = apd.rename(columns={k: v for k, v in rename_map.items() if k in apd.columns})
            apd.attrs['used_header_row'] = hdr
            return apd
        except Exception as e:
            last_error = e
            continue
    st.error(f"Error reading first 12 columns (attempted header rows {attempted_headers}): {last_error}")
    return None

def process_row_data(df_row):
    """Calculate cumulative distance based on Easting and Northing."""
    easting = df_row['Easting'].values
    northing = df_row['Northing'].values
    distances = np.sqrt(np.diff(easting)**2 + np.diff(northing)**2)
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
    elevations = df_row['EG'].values
    return np.vstack((cumulative_distances, elevations)).T

def calculate_line_of_best_fit(points):
    """Calculate slope and intercept using least squares."""
    if points.shape[0] < 2:
        return 0, 0
    x, y = points[:, 0], points[:, 1]
    n = len(x)
    sum_x, sum_y = np.sum(x), np.sum(y)
    sum_xy, sum_xx = np.sum(x * y), np.sum(x * x)
    denominator = n * sum_xx - sum_x**2
    if denominator == 0:
        return np.inf, 0
    m = (n * sum_xy - sum_x * sum_y) / denominator
    b = (sum_y - m * sum_x) / n
    return m, b

def refine_racking_line(ground_points, m, b, min_reveal, max_reveal):
    """Refine racking line by vertical shift."""
    top_of_pile_elevation = m * ground_points[:, 0] + b
    reveal_height_before_grading = top_of_pile_elevation - ground_points[:, 1]
    ground_adjustment = np.zeros_like(reveal_height_before_grading)

    cut_indices = np.where(reveal_height_before_grading < min_reveal)
    ground_adjustment[cut_indices] = reveal_height_before_grading[cut_indices] - min_reveal

    fill_indices = np.where(reveal_height_before_grading > max_reveal)
    ground_adjustment[fill_indices] = reveal_height_before_grading[fill_indices] - max_reveal

    adjustment_offset = 0.0
    if np.all(ground_adjustment <= 0) and np.any(ground_adjustment < 0):
        adjustment_offset = np.max(ground_adjustment)
    elif np.all(ground_adjustment >= 0) and np.any(ground_adjustment > 0):
        adjustment_offset = np.max(ground_adjustment)

    return b + adjustment_offset

def calculate_optimal_fixed_line(ground_points, min_reveal, max_reveal):
    """Find optimal racking line by fixing end piles."""
    if len(ground_points) < 3:
        reveal_height = (min_reveal + max_reveal) / 2
        first_pile_x = ground_points[0, 0]
        first_pile_top = ground_points[0, 1] + reveal_height
        last_pile_x = ground_points[-1, 0]
        last_pile_top = ground_points[-1, 1] + reveal_height
        slope = (last_pile_top - first_pile_top) / (last_pile_x - first_pile_x) if (last_pile_x - first_pile_x) != 0 else 0
        intercept = first_pile_top - slope * first_pile_x
        return slope, intercept

    best_slope, best_intercept = 0, 0
    min_total_grading = float('inf')

    for reveal_height in np.arange(min_reveal, max_reveal + 0.1, 0.1):
        first_pile_x = ground_points[0, 0]
        first_pile_top = ground_points[0, 1] + reveal_height
        last_pile_x = ground_points[-1, 0]
        last_pile_top = ground_points[-1, 1] + reveal_height
        slope = (last_pile_top - first_pile_top) / (last_pile_x - first_pile_x)
        intercept = first_pile_top - slope * first_pile_x

        intermediate_points = ground_points[1:-1]
        top_of_pile_intermediate = slope * intermediate_points[:, 0] + intercept
        reveal_before_grading = top_of_pile_intermediate - intermediate_points[:, 1]

        ground_adj = np.zeros_like(reveal_before_grading)
        cut_indices = np.where(reveal_before_grading < min_reveal)
        ground_adj[cut_indices] = reveal_before_grading[cut_indices] - min_reveal
        fill_indices = np.where(reveal_before_grading > max_reveal)
        ground_adj[fill_indices] = reveal_before_grading[fill_indices] - max_reveal

        current_total_grading = np.sum(np.abs(ground_adj))
        if current_total_grading < min_total_grading:
            min_total_grading = current_total_grading
            best_slope = slope
            best_intercept = intercept

    return best_slope, best_intercept

def calculate_earthwork_volume(ground_points, ground_adjustment, grading_width):
    """Calculate cut and fill volumes."""
    distances = np.diff(ground_points[:, 0])
    cuts = np.where(ground_adjustment < 0, ground_adjustment, 0)
    fills = np.where(ground_adjustment > 0, ground_adjustment, 0)

    cut_areas = (cuts[:-1] + cuts[1:]) / 2 * distances
    fill_areas = (fills[:-1] + fills[1:]) / 2 * distances

    total_cut_cf = np.sum(np.abs(cut_areas)) * grading_width
    total_fill_cf = np.sum(fill_areas) * grading_width

    return total_cut_cf, total_fill_cf

def visualize_optimization(ground_points, m, b, row_name="", final_ground=None, min_reveal=4.0, max_reveal=5.0, title_prefix=""):
    """Create visualization plot."""
    x_coords, y_coords = ground_points[:, 0], ground_points[:, 1]
    line_x = np.array([np.min(x_coords), np.max(x_coords)])
    line_y = m * line_x + b

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot(x_coords, y_coords, 'o-', label='Original Ground (EG)', color='brown', markersize=8, zorder=3)

    if final_ground is not None:
        ax.plot(x_coords, final_ground, 'g--', label='Final Ground (Post-Grading)', zorder=3, linewidth=2)

    ax.plot(line_x, line_y, '--', label=f'Final Racking Line ({min_reveal}-{max_reveal}ft Constraint)', 
            color='deepskyblue', linewidth=2.5, zorder=4)

    for i in range(len(ground_points)):
        pile_top_y = m * x_coords[i] + b
        pile_bottom_y = final_ground[i] if final_ground is not None else y_coords[i]
        ax.plot([x_coords[i], x_coords[i]], [pile_bottom_y, pile_top_y], 
                color='gray', linewidth=4, alpha=0.7, label='Pile' if i == 0 else "")

    title = f'{title_prefix} Optimization for {row_name}' if row_name else f'{title_prefix} Optimization'
    ax.set_title(title, fontsize=18, weight='bold')
    ax.set_xlabel('Distance Along Array Row (feet)', fontsize=12)
    ax.set_ylabel('Elevation (feet)', fontsize=12)
    ax.legend()
    fig.tight_layout()

    return fig

def recalculate_ns_grading_and_reveal(df, min_reveal, max_reveal):
    """Recalculate grading and reveal for N-S constrained results."""
    reveal_before_grading = df['Top of Pile (N-S Constrained)'] - df['EG']
    ground_adj = np.zeros_like(reveal_before_grading.values)

    cut_indices = np.where(reveal_before_grading < min_reveal)
    ground_adj[cut_indices] = reveal_before_grading.iloc[cut_indices] - min_reveal

    fill_indices = np.where(reveal_before_grading > max_reveal)
    ground_adj[fill_indices] = reveal_before_grading.iloc[fill_indices] - max_reveal

    df['Ground Adj (N-S Constrained)'] = ground_adj

    if 'EG' in df.columns:
        df['Finished Ground (N-S Constrained)'] = df['EG'] + df['Ground Adj (N-S Constrained)']
        df['Final Reveal (N-S Constrained)'] = df['Top of Pile (N-S Constrained)'] - df['Finished Ground (N-S Constrained)']
    else:
        df['Final Reveal (N-S Constrained)'] = df['Top of Pile (N-S Constrained)'] - (df['EG'] + ground_adj)

    df['Grading Direction (N-S Constrained)'] = np.select(
        [df['Ground Adj (N-S Constrained)'] > 0.001, df['Ground Adj (N-S Constrained)'] < -0.001],
        ['Fill', 'Cut'],
        default='None'
    )

    # Auto-compute Finished Ground for any Ground Adj (TYPE) present
    if 'EG' in df.columns:
        for col in df.columns:
            if col.startswith('Ground Adj (') and col.endswith(')'):
                typ = col[len('Ground Adj ('):-1]
                fg_col = f'Finished Ground ({typ})'
                if fg_col not in df.columns:
                    df[fg_col] = df['EG'] + df[col]

    return df

# ===================================================================
# INTERACTIVE VISUALIZATION HELPERS
# ===================================================================

def compute_row_station(row_df: pd.DataFrame):
    """Return cumulative horizontal distance along the row using Easting/Northing (ft)."""
    if not {'Easting', 'Northing'}.issubset(row_df.columns):
        return np.arange(len(row_df)) * 1.0  # fallback index-based spacing
    coords = row_df[['Easting', 'Northing']].values
    if len(coords) < 2:
        return np.array([0.0])
    seg = np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))
    return np.insert(np.cumsum(seg), 0, 0.0)

def build_row_plot(row_df: pd.DataFrame, methods: list, min_reveal: float, max_reveal: float):
    """Create an interactive Plotly figure for a single row with selected methods.

    methods: list of method name strings, e.g. ['Simple LOBF','Refined LOBF']
    """
    station = compute_row_station(row_df)

    fig = go.Figure()

    # Base ground (Existing)
    pile_ids = row_df['Pile'] if 'Pile' in row_df.columns else pd.Series([None]*len(row_df))
    point_numbers = row_df['Point Number'] if 'Point Number' in row_df.columns else pd.Series([None]*len(row_df))

    if 'EG' in row_df.columns:
        fig.add_trace(go.Scatter(
            x=station,
            y=row_df['EG'],
            mode='lines+markers',
            name='Existing Ground',
            line=dict(color='saddlebrown', width=2),
            marker=dict(size=6),
            customdata=np.column_stack([pile_ids, point_numbers]),
            hovertemplate='Pile %{customdata[0]}<br>Point %{customdata[1]}<br>Station %{x:.2f} ft<br>EG %{y:.2f} ft<extra></extra>'
        ))

    method_color_map = {
        'Simple LOBF': '#1f77b4',
        'Refined LOBF': '#2ca02c',
        'Dynamic Fixed': '#ff7f0e',
        'N-S Constrained': '#d62728'
    }

    for method in methods:
        top_col = f'Top of Pile ({method})'
        fg_col = f'Finished Ground ({method})'
        ga_col = f'Ground Adj ({method})'
        reveal_col = f'Final Reveal ({method})'
        if top_col not in row_df.columns:
            continue
        color = method_color_map.get(method, None)

        # Finished ground (post grading)
        if fg_col in row_df.columns:
            fig.add_trace(go.Scatter(
                x=station,
                y=row_df[fg_col],
                mode='lines',
                name=f'Finished Ground ({method})',
                line=dict(color=color, dash='dot'),
                customdata=np.column_stack([pile_ids, point_numbers]),
                hovertemplate='Method: '+method+'<br>Pile %{customdata[0]}<br>Point %{customdata[1]}<br>Station %{x:.2f} ft<br>FG %{y:.2f} ft<extra></extra>'
            ))

        # Top of pile line
        fig.add_trace(go.Scatter(
            x=station,
            y=row_df[top_col],
            mode='lines',
            name=f'Top of Pile ({method})',
            line=dict(color=color, width=3),
            customdata=np.column_stack([
                pile_ids,
                point_numbers,
                row_df.get(f'Row Drop ({method})', pd.Series([np.nan]*len(row_df))).values
            ]),
            hovertemplate='Method: '+method+'<br>Pile %{customdata[0]}<br>Point %{customdata[1]}<br>Row Drop %{customdata[2]:.2f} ft<br>Station %{x:.2f} ft<br>Top %{y:.2f} ft<extra></extra>'
        ))

        # Reveal as vertical bars (use shapes or separate traces)
        if reveal_col in row_df.columns and fg_col in row_df.columns:
            fig.add_trace(go.Bar(
                x=station,
                y=row_df[reveal_col],
                name=f'Reveal ({method})',
                marker_color=color,
                opacity=0.25,
                customdata=np.column_stack([pile_ids, point_numbers]),
                hovertemplate='Method: '+method+'<br>Pile %{customdata[0]}<br>Point %{customdata[1]}<br>Station %{x:.2f} ft<br>Reveal %{y:.2f} ft<extra></extra>',
                showlegend=True
            ))

        # Cut / Fill shading relative to EG where we have ground adjustments
        if ga_col in row_df.columns and 'EG' in row_df.columns:
            cuts_mask = row_df[ga_col] < -0.001
            fills_mask = row_df[ga_col] > 0.001
            if cuts_mask.any():
                fig.add_trace(go.Scatter(
                    x=station[cuts_mask],
                    y=row_df['EG'][cuts_mask] + row_df[ga_col][cuts_mask],
                    mode='markers',
                    name=f'Cut Points ({method})',
                    marker=dict(color='red', symbol='triangle-down', size=10),
                    customdata=np.column_stack([pile_ids[cuts_mask], point_numbers[cuts_mask]]),
                    hovertemplate='Method: '+method+'<br>Pile %{customdata[0]}<br>Point %{customdata[1]}<br>Station %{x:.2f} ft<br>Cut Adj %{y:.2f} ft<extra></extra>'
                ))
            if fills_mask.any():
                fig.add_trace(go.Scatter(
                    x=station[fills_mask],
                    y=row_df['EG'][fills_mask] + row_df[ga_col][fills_mask],
                    mode='markers',
                    name=f'Fill Points ({method})',
                    marker=dict(color='green', symbol='triangle-up', size=10),
                    customdata=np.column_stack([pile_ids[fills_mask], point_numbers[fills_mask]]),
                    hovertemplate='Method: '+method+'<br>Pile %{customdata[0]}<br>Point %{customdata[1]}<br>Station %{x:.2f} ft<br>Fill Adj %{y:.2f} ft<extra></extra>'
                ))

    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=40),
        xaxis_title='Station Along Row (ft)',
        yaxis_title='Elevation (ft)',
        hovermode='x unified',
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='left', x=0)
    )

    # Add reveal band limits for reference
    fig.add_hrect(
        y0=min_reveal, y1=max_reveal,
        fillcolor='rgba(100,100,255,0.05)', line_width=0,
        annotation_text='Reveal Band', annotation_position='top left'
    )

    return fig

def build_slope_hist(results_df: pd.DataFrame, methods: list):
    """Histogram / distribution of row slopes (signed) per method."""
    records = []
    for method in methods:
        drop_col = f'Row Drop ({method})'
        slope_col = f'Row Slope (%) ({method})'
        if drop_col not in results_df.columns:
            continue
        # Use one representative row entry (groupby Row first entry)
        grouped = results_df.groupby('Row').first()
        if slope_col in grouped.columns:
            for val in grouped[slope_col].dropna().values:
                records.append({'Method': method, 'Slope %': val})
    if not records:
        return None
    dist_df = pd.DataFrame(records)
    fig = px.histogram(dist_df, x='Slope %', color='Method', nbins=30, marginal='box', opacity=0.7,
                       title='Row Slope Distribution (%)')
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=40))
    return fig

def fit_plane(x, y, z):
    A = np.column_stack([x, y, np.ones_like(x)])
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    a, b, c = coeffs
    return a, b, c

def build_3d_terrain_with_planes(df: pd.DataFrame, methods: list, sample_limit: int = 15000):
    required_cols = {'Easting', 'Northing', 'EG'}
    if not required_cols.issubset(df.columns):
        return None

    # Use one record per pile (group first) for speed
    base = df.groupby(['Row', 'Pile']).first().reset_index() if {'Row','Pile'}.issubset(df.columns) else df.copy()
    if len(base) > sample_limit:
        base = base.sample(sample_limit, random_state=42)

    fig = go.Figure()

    # Existing Ground point cloud
    fig.add_trace(go.Scatter3d(
        x=base['Easting'], y=base['Northing'], z=base['EG'],
        mode='markers', name='Existing Ground',
        marker=dict(size=3, color=base['EG'], colorscale='Earth', opacity=0.7),
        hovertemplate=(
            'Row %{customdata[0]}<br>Pile %{customdata[1]}<br>'
            'Easting %{x:.2f}<br>Northing %{y:.2f}<br>EG %{z:.2f}<extra></extra>'
        ),
        customdata=np.stack([
            base['Row'] if 'Row' in base.columns else np.full(len(base), ''),
            base['Pile'] if 'Pile' in base.columns else np.full(len(base), '')
        ], axis=-1)
    ))

    xmin, xmax = base['Easting'].min(), base['Easting'].max()
    ymin, ymax = base['Northing'].min(), base['Northing'].max()
    xgrid = np.linspace(xmin, xmax, 25)
    ygrid = np.linspace(ymin, ymax, 25)
    Xg, Yg = np.meshgrid(xgrid, ygrid)

    color_cycle = {
        'Simple LOBF': '#1f77b4',
        'Refined LOBF': '#2ca02c',
        'Dynamic Fixed': '#ff7f0e',
        'N-S Constrained': '#d62728'
    }

    for method in methods:
        top_col = f'Top of Pile ({method})'
        if top_col not in df.columns:
            continue
        tops = df.groupby(['Row', 'Pile']).first().reset_index() if {'Row','Pile'}.issubset(df.columns) else df
        method_points = tops[[c for c in ['Easting','Northing', top_col] if c in tops.columns]].dropna()
        if method_points.empty:
            continue
        x = method_points['Easting'].values
        y = method_points['Northing'].values
        z = method_points[top_col].values
        a, b, c = fit_plane(x, y, z)
        Zg = a * Xg + b * Yg + c

        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers', name=f'Top Points ({method})',
            marker=dict(size=3, color=color_cycle.get(method, '#444'), opacity=0.85),
            hovertemplate=(
                f'Method: {method}<br>'
                'Easting %{x:.2f}<br>Northing %{y:.2f}<br>Top %{z:.2f}<extra></extra>'
            )
        ))

        fig.add_trace(go.Surface(
            x=Xg, y=Yg, z=Zg,
            name=f'Plane ({method})',
            showscale=False, opacity=0.35,
            surfacecolor=np.full_like(Zg, list(color_cycle.values())[0])
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Easting (ft)',
            yaxis_title='Northing (ft)',
            zaxis_title='Elevation (ft)'
        ),
        margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation='h', y=1.02, x=0)
    )
    return fig

# ===================================================================
# MAIN APPLICATION
# ===================================================================

if uploaded_file is not None:
    st.success("✓ File uploaded successfully!")

    # List sheets for convenience (lazy load once)
    try:
        xls = pd.ExcelFile(uploaded_file)
        if sheet_name not in xls.sheet_names:
            st.warning(f"Sheet '{sheet_name}' not found. Available sheets: {', '.join(xls.sheet_names)}")
        # Reset pointer so subsequent pd.read_excel works
        uploaded_file.seek(0)
    except Exception as e:
        st.error(f"Could not enumerate sheets: {e}")

    # Load data using the adaptive 12-column reader
    with st.spinner("Loading data..."):
        raw_data = read_array_pile_first_12(uploaded_file.getvalue(), sheet=sheet_name, header_row=header_row_index)

    # Diagnostics: show columns preview even if Row missing
    if raw_data is not None:
        # ================= Column Standardization & Mapping =================
        def auto_map_columns(df):
            """Attempt to map various header variants to required canonical names."""
            required = {
                'Row': ['row', 'tracker row', 'row #', 'row number', 'r'],
                'Pile': ['pile', 'pile #', 'pile number', 'p', 'post', 'post #'],
                'Easting': ['easting', 'east', 'easting (ft)', 'x', 'x coord', 'x-coordinate'],
                'Northing': ['northing', 'north', 'northing (ft)', 'y', 'y coord', 'y-coordinate'],
                'EG': ['eg', 'existing grade', 'existing ground', 'existing elev', 'existing elevation', 'ground elev', 'elev', 'elevation', 'grade']
            }
            mapping = {}
            lowered = {c.lower().strip(): c for c in df.columns}
            # Exact direct matches first
            for canon in required.keys():
                if canon in df.columns:
                    mapping[canon] = canon
            # Synonym / loose contains fallback
            for canon, variants in required.items():
                if canon in mapping:
                    continue
                for var in variants:
                    # direct lower exact
                    if var in lowered:
                        mapping[canon] = lowered[var]
                        break
                if canon in mapping:
                    continue
                # contains search
                for key_lc, orig in lowered.items():
                    if any(v in key_lc for v in variants):
                        mapping[canon] = orig
                        break
            return mapping

        # Initialize session state for manual mapping
        if 'column_mapping' not in st.session_state:
            st.session_state['column_mapping'] = {}

        auto_mapping = auto_map_columns(raw_data)

        # Merge auto mapping with existing session (manual overrides persist)
        effective_mapping = {**auto_mapping, **st.session_state['column_mapping']}

        with st.expander("🧭 Column Mapping (Adjust if needed)", expanded=False):
            st.caption("The app tries to auto-detect required columns. Adjust manually if detection is incorrect.")
            col_options = list(raw_data.columns)
            updated_mapping = {}
            map_cols = st.columns(5)
            canon_names = ['Row', 'Pile', 'Easting', 'Northing', 'EG']
            for i, canon in enumerate(canon_names):
                with map_cols[i]:
                    default_val = effective_mapping.get(canon, None)
                    if default_val not in col_options:
                        default_val = None
                    sel = st.selectbox(canon, options=['-- Select --'] + col_options,
                                       index=(col_options.index(default_val) + 1) if default_val in col_options else 0,
                                       key=f"map_{canon}")
                    if sel != '-- Select --':
                        updated_mapping[canon] = sel
            # Update session state if changed
            if updated_mapping != st.session_state['column_mapping']:
                st.session_state['column_mapping'] = updated_mapping
                effective_mapping = {**auto_mapping, **updated_mapping}
            # Show final mapping
            st.write({k: effective_mapping.get(k, '❌ Not Mapped') for k in canon_names})

        # Create canonical columns if mapping exists
        for canon, source_col in effective_mapping.items():
            if source_col in raw_data.columns and canon != source_col:
                # Avoid overwriting existing correct columns
                if canon not in raw_data.columns:
                    raw_data[canon] = raw_data[source_col]

        with st.expander("🔎 Data Preview (First 8 Rows)", expanded=False):
            st.write(f"Using header row index: {raw_data.attrs.get('used_header_row', header_row_index)}")
            st.dataframe(raw_data.head(8))
            st.caption(f"Columns detected: {list(raw_data.columns)}")

    row_col_present = raw_data is not None and 'Row' in raw_data.columns
    pile_col_present = raw_data is not None and 'Pile' in raw_data.columns
    easting_col_present = raw_data is not None and 'Easting' in raw_data.columns
    northing_col_present = raw_data is not None and 'Northing' in raw_data.columns
    eg_col_present = raw_data is not None and 'EG' in raw_data.columns

    required_ok = all([row_col_present, pile_col_present, easting_col_present, northing_col_present, eg_col_present])

    if not required_ok:
        missing = [name for name, present in [
            ('Row', row_col_present), ('Pile', pile_col_present), ('Easting', easting_col_present),
            ('Northing', northing_col_present), ('EG', eg_col_present)
        ] if not present]
        st.warning(f"Required columns missing: {missing}. Try adjusting 'Header Row Index' or verifying the sheet name.")
        st.info("You can still adjust settings and re-upload. The Run button is disabled until required columns are found.")

    if required_ok:
        st.info(f"📊 Loaded {len(raw_data)} data points from {len(raw_data['Row'].unique())} rows")

    # Show Run button regardless (disabled if columns invalid)
    run_clicked = st.button("🚀 Run Optimization Analysis", type="primary", disabled=not required_ok)

    if run_clicked and required_ok:
            with st.spinner("Processing optimization..."):
                data_to_process = raw_data.copy()
                grouped = data_to_process.groupby('Row')
                all_summary_dfs = []

                progress_bar = st.progress(0)
                total_rows = len(grouped)

                for idx, (row_name, row_df) in enumerate(grouped):
                    ground_points = process_row_data(row_df)
                    ideal_reveal = (min_reveal_height + max_reveal_height) / 2
                    ideal_top_of_pile_points = ground_points.copy()
                    ideal_top_of_pile_points[:, 1] += ideal_reveal

                    # Simple LOBF
                    slope_simple, intercept_simple = calculate_line_of_best_fit(ideal_top_of_pile_points)
                    top_of_pile_simple = slope_simple * ground_points[:, 0] + intercept_simple
                    reveal_before_grading_simple = top_of_pile_simple - ground_points[:, 1]

                    ground_adj_simple = np.zeros_like(reveal_before_grading_simple)
                    cut_indices_simple = np.where(reveal_before_grading_simple < min_reveal_height)
                    ground_adj_simple[cut_indices_simple] = reveal_before_grading_simple[cut_indices_simple] - min_reveal_height
                    fill_indices_simple = np.where(reveal_before_grading_simple > max_reveal_height)
                    ground_adj_simple[fill_indices_simple] = reveal_before_grading_simple[fill_indices_simple] - max_reveal_height

                    final_ground_simple = ground_points[:, 1] + ground_adj_simple
                    final_reveal_simple = top_of_pile_simple - final_ground_simple

                    # Refined LOBF
                    slope_refined, intercept_base = slope_simple, intercept_simple
                    intercept_refined = refine_racking_line(ground_points, slope_refined, intercept_base, 
                                                           min_reveal_height, max_reveal_height)
                    top_of_pile_refined = slope_refined * ground_points[:, 0] + intercept_refined
                    reveal_before_grading_refined = top_of_pile_refined - ground_points[:, 1]

                    ground_adj_refined = np.zeros_like(reveal_before_grading_refined)
                    cut_indices_refined = np.where(reveal_before_grading_refined < min_reveal_height)
                    ground_adj_refined[cut_indices_refined] = reveal_before_grading_refined[cut_indices_refined] - min_reveal_height
                    fill_indices_refined = np.where(reveal_before_grading_refined > max_reveal_height)
                    ground_adj_refined[fill_indices_refined] = reveal_before_grading_refined[fill_indices_refined] - max_reveal_height

                    final_ground_refined = ground_points[:, 1] + ground_adj_refined
                    final_reveal_refined = top_of_pile_refined - final_ground_refined

                    # Dynamic Fixed
                    slope_fixed, intercept_fixed = calculate_optimal_fixed_line(ground_points, 
                                                                                min_reveal_height, max_reveal_height)
                    top_of_pile_fixed = slope_fixed * ground_points[:, 0] + intercept_fixed
                    reveal_before_grading_fixed = top_of_pile_fixed - ground_points[:, 1]

                    ground_adj_fixed = np.zeros_like(reveal_before_grading_fixed)
                    cut_indices_fixed = np.where(reveal_before_grading_fixed < min_reveal_height)
                    ground_adj_fixed[cut_indices_fixed] = reveal_before_grading_fixed[cut_indices_fixed] - min_reveal_height
                    fill_indices_fixed = np.where(reveal_before_grading_fixed > max_reveal_height)
                    ground_adj_fixed[fill_indices_fixed] = reveal_before_grading_fixed[fill_indices_fixed] - max_reveal_height

                    final_ground_fixed = ground_points[:, 1] + ground_adj_fixed
                    final_reveal_fixed = top_of_pile_fixed - final_ground_fixed

                    # Create summary DataFrame
                    summary_df = row_df.copy()

                    summary_df['Top of Pile (Simple LOBF)'] = top_of_pile_simple
                    summary_df['Ground Adj (Simple LOBF)'] = ground_adj_simple
                    summary_df['Final Reveal (Simple LOBF)'] = final_reveal_simple

                    summary_df['Top of Pile (Refined LOBF)'] = top_of_pile_refined
                    summary_df['Ground Adj (Refined LOBF)'] = ground_adj_refined
                    summary_df['Final Reveal (Refined LOBF)'] = final_reveal_refined

                    summary_df['Top of Pile (Dynamic Fixed)'] = top_of_pile_fixed
                    summary_df['Ground Adj (Dynamic Fixed)'] = ground_adj_fixed
                    summary_df['Final Reveal (Dynamic Fixed)'] = final_reveal_fixed

                    # Add Finished Ground columns
                    if 'EG' in summary_df.columns:
                        summary_df['Finished Ground (Simple LOBF)'] = summary_df['EG'] + summary_df['Ground Adj (Simple LOBF)']
                        summary_df['Finished Ground (Refined LOBF)'] = summary_df['EG'] + summary_df['Ground Adj (Refined LOBF)']
                        summary_df['Finished Ground (Dynamic Fixed)'] = summary_df['EG'] + summary_df['Ground Adj (Dynamic Fixed)']

                    all_summary_dfs.append(summary_df)

                    # Test mode visualizations
                    if test_mode and idx < 3:  # Only show first 3 rows in test mode
                        st.subheader(f"Row {row_name} Visualizations")

                        col1, col2, col3 = st.columns(3)

                        with col1:
                            fig = visualize_optimization(ground_points, slope_simple, intercept_simple, 
                                                        row_name=str(row_name), final_ground=final_ground_simple,
                                                        min_reveal=min_reveal_height, max_reveal=max_reveal_height,
                                                        title_prefix="Simple LOBF")
                            st.pyplot(fig)
                            plt.close()

                        with col2:
                            fig = visualize_optimization(ground_points, slope_refined, intercept_refined,
                                                        row_name=str(row_name), final_ground=final_ground_refined,
                                                        min_reveal=min_reveal_height, max_reveal=max_reveal_height,
                                                        title_prefix="Refined LOBF")
                            st.pyplot(fig)
                            plt.close()

                        with col3:
                            fig = visualize_optimization(ground_points, slope_fixed, intercept_fixed,
                                                        row_name=str(row_name), final_ground=final_ground_fixed,
                                                        min_reveal=min_reveal_height, max_reveal=max_reveal_height,
                                                        title_prefix="Dynamic Fixed")
                            st.pyplot(fig)
                            plt.close()

                    progress_bar.progress((idx + 1) / total_rows)

                # Combine all results
                final_results_df = pd.concat(all_summary_dfs, ignore_index=True)

                # Apply N-S constraints if enabled
                if apply_ns_constraints:
                    st.info("Applying North-South Constraints...")

                    # Determine best base method
                    total_grading_simple = np.sum(np.abs(final_results_df['Ground Adj (Simple LOBF)']))
                    total_grading_refined = np.sum(np.abs(final_results_df['Ground Adj (Refined LOBF)']))
                    total_grading_fixed = np.sum(np.abs(final_results_df['Ground Adj (Dynamic Fixed)']))

                    base_results = {
                        "Simple LOBF": total_grading_simple,
                        "Refined LOBF": total_grading_refined,
                        "Dynamic Fixed": total_grading_fixed
                    }
                    best_mode_name = min(base_results, key=base_results.get)

                    column_mapping = {
                        "Simple LOBF": "Top of Pile (Simple LOBF)",
                        "Refined LOBF": "Top of Pile (Refined LOBF)",
                        "Dynamic Fixed": "Top of Pile (Dynamic Fixed)"
                    }
                    base_col_name = column_mapping[best_mode_name]
                    final_results_df['Top of Pile (N-S Constrained)'] = final_results_df[base_col_name]

                    # Initialize Adjacent Row columns
                    final_results_df['Adjacent Row to North'] = np.nan
                    final_results_df['Adjacent Row to South'] = np.nan

                    # Simplified N-S constraint logic
                    all_rows_in_data = set(final_results_df['Row'].unique())

                    # Group by Easting to find adjacent rows
                    final_results_df['EastingGroup'] = (final_results_df['Easting'] / 100) * 100

                    for easting_group, group_df in final_results_df.groupby('EastingGroup'):
                        row_avg_coords = group_df.groupby('Row').agg(
                            avg_northing=('Northing', 'mean'),
                            avg_easting=('Easting', 'mean')
                        ).sort_values('avg_northing', ascending=False)

                        sorted_row_names = list(row_avg_coords.index)
                        processed_rows = set()

                        for north_row_name in sorted_row_names:
                            if north_row_name in processed_rows:
                                continue

                            north_coords = row_avg_coords.loc[north_row_name]
                            potential_south_candidates = row_avg_coords[
                                (row_avg_coords.index != north_row_name) &
                                (row_avg_coords['avg_northing'] < north_coords['avg_northing']) &
                                ((row_avg_coords['avg_easting'] - north_coords['avg_easting']).abs() < 5)
                            ]

                            unprocessed_candidates = potential_south_candidates[
                                ~potential_south_candidates.index.isin(processed_rows)
                            ]

                            if unprocessed_candidates.empty:
                                continue

                            distances = {}
                            north_row_df = final_results_df[final_results_df['Row'] == north_row_name]
                            south_post_of_north_row = north_row_df.loc[north_row_df['Northing'].idxmin()]

                            for candidate_row_name in unprocessed_candidates.index:
                                candidate_data = final_results_df[final_results_df['Row'] == candidate_row_name]
                                north_post_of_candidate_row = candidate_data.loc[candidate_data['Northing'].idxmax()]

                                dist_northing = south_post_of_north_row['Northing'] - north_post_of_candidate_row['Northing']
                                dist_easting = south_post_of_north_row['Easting'] - north_post_of_candidate_row['Easting']
                                distances[candidate_row_name] = np.sqrt(dist_northing**2 + dist_easting**2)

                            if not distances:
                                continue

                            south_row_name = min(distances, key=distances.get)

                            # Apply constraints
                            north_row_data = final_results_df[final_results_df['Row'] == north_row_name].copy()
                            south_row_data = final_results_df[final_results_df['Row'] == south_row_name].copy()

                            south_post_of_north_row = north_row_data.loc[north_row_data['Northing'].idxmin()]
                            north_post_of_south_row = south_row_data.loc[south_row_data['Northing'].idxmax()]

                            northing_diff = abs(south_post_of_north_row['Northing'] - north_post_of_south_row['Northing'])

                            adjustment_made = False
                            new_north_top, new_south_top = None, None

                            if northing_diff < ns_northing_diff_min_ft:
                                top_n = south_post_of_north_row['Top of Pile (N-S Constrained)']
                                top_s = north_post_of_south_row['Top of Pile (N-S Constrained)']

                                reveal_north = top_n - south_post_of_north_row['EG']
                                reveal_south = top_s - north_post_of_south_row['EG']
                                avg_reveal = (reveal_north + reveal_south) / 2
                                target_reveal = np.clip(avg_reveal, min_reveal_height, max_reveal_height)

                                # Start by targeting equal reveal on both facing posts.
                                eg_n = south_post_of_north_row['EG']
                                eg_s = north_post_of_south_row['EG']

                                new_north_top = eg_n + target_reveal
                                new_south_top = eg_s + target_reveal

                                # If requested, apply a tighter close-spaced top-of-pile tolerance.
                                tol_close = ns_top_of_pile_tolerance_close_ft
                                if tol_close is not None and tol_close >= 0.0 and abs(new_north_top - new_south_top) > tol_close:
                                    # Try to bring both tops near a common elevation while respecting reveal limits.
                                    avg_top = (new_north_top + new_south_top) / 2

                                    n_min = eg_n + min_reveal_height
                                    n_max = eg_n + max_reveal_height
                                    s_min = eg_s + min_reveal_height
                                    s_max = eg_s + max_reveal_height

                                    new_north_top = float(np.clip(avg_top, n_min, n_max))
                                    new_south_top = float(np.clip(avg_top, s_min, s_max))

                                    # Second pass: if still out of tolerance (due to clipping), compress the gap.
                                    if abs(new_north_top - new_south_top) > tol_close:
                                        if new_north_top > new_south_top:
                                            new_north_top = float(np.clip(new_south_top + tol_close, n_min, n_max))
                                        else:
                                            new_south_top = float(np.clip(new_north_top + tol_close, s_min, s_max))

                                # Mark adjustment only if we actually changed something.
                                if (not np.isclose(new_north_top, top_n)) or (not np.isclose(new_south_top, top_s)):
                                    adjustment_made = True

                            elif ns_northing_diff_min_ft <= northing_diff < ns_northing_diff_max_ft:
                                top_n = south_post_of_north_row['Top of Pile (N-S Constrained)']
                                top_s = north_post_of_south_row['Top of Pile (N-S Constrained)']
                                reveal_n = top_n - south_post_of_north_row['EG']
                                reveal_s = top_s - north_post_of_south_row['EG']

                                ideal_reveal = (min_reveal_height + max_reveal_height) / 2

                                new_top_n_tentative = top_n
                                if not (min_reveal_height <= reveal_n <= max_reveal_height):
                                    new_top_n_tentative = south_post_of_north_row['EG'] + ideal_reveal

                                new_top_s_tentative = top_s
                                if not (min_reveal_height <= reveal_s <= max_reveal_height):
                                    new_top_s_tentative = north_post_of_south_row['EG'] + ideal_reveal

                                if abs(new_top_n_tentative - new_top_s_tentative) > ns_top_of_pile_tolerance_ft:
                                    new_north_top = new_top_n_tentative
                                    new_south_top = new_top_s_tentative
                                else:
                                    if (top_n - top_s) > ns_top_of_pile_tolerance_ft:
                                        new_north_top = top_s + ns_top_of_pile_tolerance_ft
                                    elif (top_s - top_n) > ns_top_of_pile_tolerance_ft:
                                        new_south_top = top_n + ns_top_of_pile_tolerance_ft

                                if (new_north_top is not None and not np.isclose(new_north_top, top_n)) or                                    (new_south_top is not None and not np.isclose(new_south_top, top_s)):
                                    adjustment_made = True

                            if adjustment_made:
                                if new_north_top is not None:
                                    north_gp = process_row_data(north_row_data)
                                    pivot_post = north_row_data.loc[north_row_data['Northing'].idxmax()]
                                    pivot_point_idx = north_row_data.index.get_loc(pivot_post.name)

                                    pivot_x = north_gp[pivot_point_idx, 0]
                                    pivot_y_top = pivot_post['Top of Pile (N-S Constrained)']

                                    adjusted_post_idx = north_row_data.index.get_loc(south_post_of_north_row.name)
                                    adjusted_x = north_gp[adjusted_post_idx, 0]

                                    m = (new_north_top - pivot_y_top) / (adjusted_x - pivot_x) if (adjusted_x - pivot_x) != 0 else 0
                                    b = pivot_y_top - m * pivot_x

                                    final_results_df.loc[final_results_df['Row'] == north_row_name, 'Top of Pile (N-S Constrained)'] =                                         m * north_gp[:, 0] + b

                                if new_south_top is not None:
                                    south_gp = process_row_data(south_row_data)
                                    pivot_post = south_row_data.loc[south_row_data['Northing'].idxmin()]
                                    pivot_point_idx = south_row_data.index.get_loc(pivot_post.name)

                                    pivot_x = south_gp[pivot_point_idx, 0]
                                    pivot_y_top = pivot_post['Top of Pile (N-S Constrained)']

                                    adjusted_post_idx = south_row_data.index.get_loc(north_post_of_south_row.name)
                                    adjusted_x = south_gp[adjusted_post_idx, 0]

                                    m = (pivot_y_top - new_south_top) / (pivot_x - adjusted_x) if (pivot_x - adjusted_x) != 0 else 0
                                    b = new_south_top - m * adjusted_x

                                    final_results_df.loc[final_results_df['Row'] == south_row_name, 'Top of Pile (N-S Constrained)'] =                                         m * south_gp[:, 0] + b

                                final_results_df.loc[final_results_df['Row'] == north_row_name, 'Adjacent Row to South'] = south_row_name
                                final_results_df.loc[final_results_df['Row'] == south_row_name, 'Adjacent Row to North'] = north_row_name

                            processed_rows.add(north_row_name)
                            processed_rows.add(south_row_name)

                    # Recalculate with N-S constraints
                    final_results_df = recalculate_ns_grading_and_reveal(final_results_df, 
                                                                         min_reveal_height, max_reveal_height)

                # Display summary statistics
                st.success("✓ Analysis Complete!")
                st.header("📊 Overall Project Summary")

                methods = ['Simple LOBF', 'Refined LOBF', 'Dynamic Fixed']
                if apply_ns_constraints:
                    methods.append('N-S Constrained')

                summary_results = {}
                summary_data = []

                for method in methods:
                    ground_adj_col = f'Ground Adj ({method})'
                    reveal_col = f'Final Reveal ({method})'

                    total_cut_volume = 0
                    total_fill_volume = 0
                    total_pile_length = np.sum(final_results_df[reveal_col])

                    for row_name, row_data in final_results_df.groupby('Row'):
                        row_ground_points = process_row_data(row_data)
                        ground_adjustments = row_data[ground_adj_col].values
                        cut_vol, fill_vol = calculate_earthwork_volume(row_ground_points, ground_adjustments, 
                                                                       grading_width_ft)
                        total_cut_volume += cut_vol
                        total_fill_volume += fill_vol

                    total_grading_volume = total_cut_volume + total_fill_volume
                    summary_results[method] = {
                        'total_grading': total_grading_volume,
                        'pile_length': total_pile_length
                    }

                    summary_data.append({
                        'Method': method,
                        'Cut Volume (ft³)': f"{total_cut_volume:.2f}",
                        'Fill Volume (ft³)': f"{total_fill_volume:.2f}",
                        'Total Grading (ft³)': f"{total_grading_volume:.2f}",
                        'Pile Length (ft)': f"{total_pile_length:.2f}"
                    })

                # Display summary table
                summary_table_df = pd.DataFrame(summary_data)
                st.dataframe(summary_table_df, use_container_width=True)

                # Determine optimal method
                optimal_method = min(summary_results.items(), 
                                    key=lambda x: (x[1]['total_grading'], x[1]['pile_length']))[0]
                st.success(f"🏆 **Optimal Method:** {optimal_method}")

                # Store results in session state
                st.session_state['final_results_df'] = final_results_df
                st.session_state['optimal_method'] = optimal_method
                st.session_state['apply_ns_constraints'] = apply_ns_constraints

                # Interactive visualization section (only in test mode)
                if test_mode:
                    st.header("📈 Interactive Visualizations")
                    if not PLOTLY_AVAILABLE:
                        st.warning("Plotly not installed. Run 'pip install plotly' in your environment to enable interactive charts.")
                    else:
                        viz_methods = []
                        base_methods = ['Simple LOBF', 'Refined LOBF', 'Dynamic Fixed']
                        for m in base_methods:
                            if f'Top of Pile ({m})' in final_results_df.columns:
                                viz_methods.append(m)
                        if apply_ns_constraints and 'Top of Pile (N-S Constrained)' in final_results_df.columns:
                            viz_methods.append('N-S Constrained')

                        col_v1, col_v2, col_v3 = st.columns([2,2,2])
                        with col_v1:
                            all_rows = sorted(final_results_df['Row'].unique())
                            selected_rows = st.multiselect("Select Rows", all_rows, default=all_rows[:1])
                        with col_v2:
                            selected_methods = st.multiselect("Methods", viz_methods, default=viz_methods)
                        with col_v3:
                            show_slope_dist = st.checkbox("Show Slope Distribution", value=True)
                        show_3d = st.checkbox("Show 3D Terrain & Planes", value=False)

                        if selected_rows and selected_methods:
                            for row in selected_rows:
                                row_df = final_results_df[final_results_df['Row'] == row]
                                st.markdown(f"#### Row {row}")
                                try:
                                    fig_row = build_row_plot(row_df, selected_methods, min_reveal_height, max_reveal_height)
                                    st.plotly_chart(fig_row, use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error creating row plot for Row {row}: {e}")
                        elif not selected_rows:
                            st.info("Select at least one row to display.")
                        elif not selected_methods:
                            st.info("Select at least one method to display row profile.")

                        if show_slope_dist:
                            try:
                                slope_fig = build_slope_hist(final_results_df[final_results_df['Row'].isin(selected_rows)], selected_methods)
                                if slope_fig:
                                    st.plotly_chart(slope_fig, use_container_width=True)
                            except Exception as e:
                                st.error(f"Error creating slope distribution: {e}")

                        if show_3d:
                            try:
                                terrain_fig = build_3d_terrain_with_planes(final_results_df[final_results_df['Row'].isin(selected_rows)], selected_methods)
                                if terrain_fig:
                                    st.plotly_chart(terrain_fig, use_container_width=True)
                                else:
                                    st.info("3D plot unavailable (missing columns or no data).")
                            except Exception as e:
                                st.error(f"Error creating 3D terrain: {e}")

                # Display data preview
                st.subheader("📋 Results Preview")
                st.dataframe(final_results_df.head(20), use_container_width=True)

    # Download buttons section - moved outside run_clicked to persist after rerun
    if 'final_results_df' in st.session_state and st.session_state['final_results_df'] is not None:
        final_results_df = st.session_state['final_results_df']
        apply_ns_constraints = st.session_state.get('apply_ns_constraints', True)
        
        # Download buttons
        st.subheader("💾 Download Results")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Excel export - Keep all columns like original script
            output = io.BytesIO()

            # Prepare export dataframe with all columns
            base_columns = ['Row', 'Pile', 'Easting', 'Northing', 'EG']

            # Add Point Number if it exists
            if 'Point Number' in final_results_df.columns:
                base_columns.insert(0, 'Point Number')

            calc_columns = [
                'Top of Pile (Simple LOBF)', 'Ground Adj (Simple LOBF)', 'Final Reveal (Simple LOBF)', 'Finished Ground (Simple LOBF)',
                'Top of Pile (Refined LOBF)', 'Ground Adj (Refined LOBF)', 'Final Reveal (Refined LOBF)', 'Finished Ground (Refined LOBF)',
                'Top of Pile (Dynamic Fixed)', 'Ground Adj (Dynamic Fixed)', 'Final Reveal (Dynamic Fixed)', 'Finished Ground (Dynamic Fixed)',
            ]

            if apply_ns_constraints:
                calc_columns.extend([
                    'Top of Pile (N-S Constrained)', 'Ground Adj (N-S Constrained)', 
                    'Final Reveal (N-S Constrained)', 'Finished Ground (N-S Constrained)',
                    'Grading Direction (N-S Constrained)', 'Adjacent Row to North', 'Adjacent Row to South'
                ])

            # Remove EastingGroup if present
            if 'EastingGroup' in final_results_df.columns:
                final_results_df_export = final_results_df.drop('EastingGroup', axis=1)
            else:
                final_results_df_export = final_results_df.copy()

            # Select only existing columns
            export_columns = [col for col in base_columns + calc_columns if col in final_results_df_export.columns]
            export_df = final_results_df_export[export_columns].copy()

            # ================= Row-to-Row Drop Calculations =================
            # Definition: For each Row & method, Row Drop = last pile top - first pile top (signed)
            # Also compute Row Slope (%) = (Row Drop / horizontal length)*100
            method_top_cols = {
                'Simple LOBF': 'Top of Pile (Simple LOBF)',
                'Refined LOBF': 'Top of Pile (Refined LOBF)',
                'Dynamic Fixed': 'Top of Pile (Dynamic Fixed)'
            }
            if apply_ns_constraints and 'Top of Pile (N-S Constrained)' in export_df.columns:
                method_top_cols['N-S Constrained'] = 'Top of Pile (N-S Constrained)'

            # Initialize columns (will reposition later)
            for method_name in method_top_cols.keys():
                export_df[f'Row Drop ({method_name})'] = np.nan
                export_df[f'Row Slope (%) ({method_name})'] = np.nan

            drop_summary_records = []

            for row_name, grp in export_df.groupby('Row'):
                # Distance ordering
                if {'Easting', 'Northing'}.issubset(grp.columns):
                    coords = grp[['Easting', 'Northing']].values
                    dists = np.insert(np.cumsum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))), 0, 0)
                    order_idx = np.argsort(dists)
                    ordered_index = grp.index[order_idx]
                    horizontal_len = dists[order_idx[-1]] - dists[order_idx[0]] if len(dists) > 1 else 0
                else:
                    ordered_index = grp.index
                    horizontal_len = len(grp) - 1 if len(grp) > 1 else 0

                record = {'Row': row_name, 'Horizontal Length (ft)': horizontal_len}

                for method_name, top_col in method_top_cols.items():
                    if top_col not in grp.columns:
                        continue
                    first_top = grp.loc[ordered_index[0], top_col]
                    last_top = grp.loc[ordered_index[-1], top_col]
                    row_drop = last_top - first_top
                    slope_pct = (row_drop / horizontal_len * 100) if horizontal_len else 0.0
                    drop_col = f'Row Drop ({method_name})'
                    slope_col = f'Row Slope (%) ({method_name})'
                    export_df.loc[grp.index, drop_col] = row_drop
                    export_df.loc[grp.index, slope_col] = slope_pct
                    record[drop_col] = row_drop
                    record[slope_col] = slope_pct
                drop_summary_records.append(record)

            row_drop_summary_df = pd.DataFrame(drop_summary_records).sort_values('Row')

            # Reorder columns to place drop & slope right after each Top of Pile column
            def reordered_columns(df: pd.DataFrame):
                base_set = set(base_columns)
                method_order = ['Simple LOBF', 'Refined LOBF', 'Dynamic Fixed']
                if apply_ns_constraints and 'Top of Pile (N-S Constrained)' in df.columns:
                    method_order.append('N-S Constrained')
                method_column_groups = []
                for m in method_order:
                    top_c = f'Top of Pile ({m})'
                    drop_c = f'Row Drop ({m})'
                    slope_c = f'Row Slope (%) ({m})'
                    ga_c = f'Ground Adj ({m})'
                    fr_c = f'Final Reveal ({m})'
                    fg_c = f'Finished Ground ({m})'
                    group = [c for c in [top_c, drop_c, slope_c, ga_c, fr_c, fg_c] if c in df.columns]
                    method_column_groups.extend(group)
                ordered = [c for c in base_columns if c in df.columns] + method_column_groups
                # Append any remaining columns not yet included
                remaining = [c for c in df.columns if c not in ordered]
                return ordered + remaining

            export_df = export_df[reordered_columns(export_df)]
            # ================= End Row-to-Row Drop Calculations =================

            # ================= Best Method & All Columns (when N-S unchecked) =================
            if not apply_ns_constraints:
                # Initialize columns
                export_df['Best_Method'] = ''
                export_df['Best_Top_of_Pile'] = np.nan
                export_df['Best_Ground_Adj'] = np.nan
                export_df['Best_Final_Reveal'] = np.nan
                export_df['Best_Finished_Ground'] = np.nan
                
                # Calculate total grading for each row and each method
                for row_name, grp in export_df.groupby('Row'):
                    # Calculate total ground adjustment (absolute sum) for each method
                    method_grading = {}
                    for method in ['Simple LOBF', 'Refined LOBF', 'Dynamic Fixed']:
                        ground_adj_col = f'Ground Adj ({method})'
                        if ground_adj_col in grp.columns:
                            ground_adjustments = grp[ground_adj_col].values
                            total_grading = np.sum(np.abs(ground_adjustments))
                            method_grading[method] = total_grading
                    
                    # Find best method (minimum total grading)
                    if method_grading:
                        best_method = min(method_grading.items(), key=lambda x: x[1])
                        best_method_name = best_method[0]
                        
                        # Get all column names for the best method
                        best_top_col = f'Top of Pile ({best_method_name})'
                        best_ground_adj_col = f'Ground Adj ({best_method_name})'
                        best_reveal_col = f'Final Reveal ({best_method_name})'
                        best_fg_col = f'Finished Ground ({best_method_name})'
                        
                        # Assign best method name and all its values to this row
                        export_df.loc[grp.index, 'Best_Method'] = best_method_name
                        
                        if best_top_col in grp.columns:
                            export_df.loc[grp.index, 'Best_Top_of_Pile'] = grp[best_top_col].values
                        
                        if best_ground_adj_col in grp.columns:
                            export_df.loc[grp.index, 'Best_Ground_Adj'] = grp[best_ground_adj_col].values
                        
                        if best_reveal_col in grp.columns:
                            export_df.loc[grp.index, 'Best_Final_Reveal'] = grp[best_reveal_col].values
                        
                        if best_fg_col in grp.columns:
                            export_df.loc[grp.index, 'Best_Finished_Ground'] = grp[best_fg_col].values
            # ================= End Best Method & All Columns =================

            # ================= Row Summary Sheet =================
            # Row number, number of piles, 1st pile top elevation, last pile top elevation, best method name.
            # Best method name must reflect whether N-S constraints were applied.
            row_summary_records = []

            # When N-S constraints are enabled, the "best" method for reporting is the constrained result.
            ns_best_method_name = None
            if apply_ns_constraints and 'Top of Pile (N-S Constrained)' in export_df.columns:
                ns_best_method_name = 'N-S Constrained'

            for row_name, grp in export_df.groupby('Row'):
                # Determine ordering along the row for first/last pile
                if {'Easting', 'Northing'}.issubset(grp.columns) and len(grp) > 1:  
                    coords = grp[['Easting', 'Northing']].values
                    dists = np.insert(np.cumsum(np.sqrt(np.sum(np.diff(coords, axis=0)**2, axis=1))), 0, 0)
                    order_idx = np.argsort(dists)
                    ordered_index = grp.index[order_idx]
                else:
                    ordered_index = grp.index

                num_piles = grp['Pile'].nunique() if 'Pile' in grp.columns else len(grp)

                if ns_best_method_name is not None:
                    best_method_name = ns_best_method_name
                else:
                    # Fall back to per-row best method (computed above when N-S is unchecked)
                    best_method_name = ''
                    if 'Best_Method' in grp.columns:
                        bm = grp['Best_Method'].iloc[0]
                        if isinstance(bm, str) and bm.strip():
                            best_method_name = bm.strip()

                top_col = f'Top of Pile ({best_method_name})' if best_method_name else None

                first_top = np.nan
                last_top = np.nan
                if top_col and top_col in grp.columns and len(ordered_index) > 0:
                    first_top = grp.loc[ordered_index[0], top_col]
                    last_top = grp.loc[ordered_index[-1], top_col]

                row_summary_records.append({
                    'Row': row_name,
                    'Number of Piles': int(num_piles) if pd.notna(num_piles) else num_piles,
                    '1st Pile Top of Elevation': first_top,
                    'Last Pile Top of Elevation': last_top,
                    'Best Method': best_method_name
                })

            row_summary_df = pd.DataFrame(row_summary_records).sort_values('Row') if row_summary_records else pd.DataFrame(
                columns=['Row', 'Number of Piles', '1st Pile Top of Elevation', 'Last Pile Top of Elevation', 'Best Method']
            )
            # ================= End Row Summary Sheet =================

            # Round numerical columns
            numerical_cols = [col for col in export_df.columns 
                             if export_df[col].dtype in ['float64', 'int64'] and col not in ['Pile', 'Point Number']]
            export_df[numerical_cols] = export_df[numerical_cols].round(2)

            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                # Round selected numeric metric columns for readability (after reordering)
                for col in export_df.columns:
                    if export_df[col].dtype in [float, int]:
                        export_df[col] = export_df[col].round(4) if 'Slope' in col else export_df[col].round(2)
                export_df.to_excel(writer, sheet_name='Optimization Results', index=False)

                # Row Summary sheet
                if not row_summary_df.empty:
                    rs_out = row_summary_df.copy()
                    for col in ['1st Pile Top of Elevation', 'Last Pile Top of Elevation']:
                        if col in rs_out.columns:
                            rs_out[col] = pd.to_numeric(rs_out[col], errors='coerce').round(2)
                    rs_out.to_excel(writer, sheet_name='Row Summary', index=False)

                # Row Drops summary sheet (signed drop + slope only)
                if not row_drop_summary_df.empty:
                    # Round for summary
                    summary_df_out = row_drop_summary_df.copy()
                    for col in summary_df_out.columns:
                        if summary_df_out[col].dtype in [float, int]:
                            summary_df_out[col] = summary_df_out[col].round(4) if 'Slope' in col else summary_df_out[col].round(2)
                    summary_df_out.to_excel(writer, sheet_name='Row Drops', index=False)
            output.seek(0)

            st.download_button(
                label="📥 Download Excel Results",
                data=output,
                file_name="optimization_results_comparison.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        with col2:
            # Cut/Fill CSV - Available for both N-S constrained and unconstrained
            point_number_col = 'Point Number' if 'Point Number' in final_results_df.columns else 'Pile'
            
            if apply_ns_constraints and 'Ground Adj (N-S Constrained)' in final_results_df.columns:
                # N-S Constrained version
                cut_fill_df = pd.DataFrame({
                    'Point number': final_results_df[point_number_col],
                    'Northing': final_results_df['Northing'].round(2),
                    'Easting': final_results_df['Easting'].round(2),
                    'Ground Adj (N-S Constrained)': final_results_df['Ground Adj (N-S Constrained)'].round(2),
                    'Grading Direction (N-S Constrained)': final_results_df['Grading Direction (N-S Constrained)']
                })
            else:
                # Best Method version (when N-S unchecked)
                # Add grading direction for best method
                grading_direction = np.select(
                    [export_df['Best_Ground_Adj'] > 0.001, export_df['Best_Ground_Adj'] < -0.001],
                    ['Fill', 'Cut'],
                    default='None'
                )
                
                cut_fill_df = pd.DataFrame({
                    'Point number': export_df[point_number_col],
                    'Northing': export_df['Northing'].round(2),
                    'Easting': export_df['Easting'].round(2),
                    'Best_Method': export_df['Best_Method'],
                    'Ground Adj (Best Method)': export_df['Best_Ground_Adj'].round(2),
                    'Grading Direction (Best Method)': grading_direction
                })

            st.download_button(
                label="📥 Download Cut_Fill.csv",
                data=cut_fill_df.to_csv(index=False),
                file_name="Cut_Fill.csv",
                mime="text/csv"
            )

        with col3:
            # FG Surface CSV - Available for both N-S constrained and unconstrained
            point_number_col = 'Point Number' if 'Point Number' in final_results_df.columns else 'Pile'
            
            if apply_ns_constraints and 'Finished Ground (N-S Constrained)' in final_results_df.columns:
                # N-S Constrained version
                fg_surface_df = pd.DataFrame({
                    'Point Number': final_results_df[point_number_col],
                    'Northing': final_results_df['Northing'].round(2),
                    'Easting': final_results_df['Easting'].round(2),
                    'Finished Ground (N-S Constrained)': final_results_df['Finished Ground (N-S Constrained)'].round(2),
                    'Grading Direction (N-S Constrained)': final_results_df['Grading Direction (N-S Constrained)']
                })
            else:
                # Best Method version (when N-S unchecked)
                grading_direction = np.select(
                    [export_df['Best_Ground_Adj'] > 0.001, export_df['Best_Ground_Adj'] < -0.001],
                    ['Fill', 'Cut'],
                    default='None'
                )
                
                fg_surface_df = pd.DataFrame({
                    'Point Number': export_df[point_number_col],
                    'Northing': export_df['Northing'].round(2),
                    'Easting': export_df['Easting'].round(2),
                    'Best_Method': export_df['Best_Method'],
                    'Finished Ground (Best Method)': export_df['Best_Finished_Ground'].round(2),
                    'Grading Direction (Best Method)': grading_direction
                })

            st.download_button(
                label="📥 Download FG_Surface.csv",
                data=fg_surface_df.to_csv(index=False),
                file_name="FG_Surface.csv",
                mime="text/csv"
            )

else:
    st.info("👈 Please upload an Excel file to begin the analysis")

    # Display instructions
    st.markdown("""
    ### 📖 Instructions

    1. **Upload your Excel file** containing Array Pile Data (typically named 'Grading.xlsx')
    2. **Configure parameters** in the sidebar:
       - Sheet name (default: "Array Pile Data")
       - Min/Max reveal heights
       - Grading width
       - North-South constraints
    3. **Click "Run Optimization Analysis"** to process your data
    4. **Review results** and download output files

    ### 📋 Required Excel Format

    Your Excel file must contain a sheet with at least the first 12 columns including:
    - Point Number (Column 0)
    - Row (Column 3)
    - Pile (Column 4)
    - Northing (Column 9)
    - Easting (Column 10)
    - EG (Column 11)

    ### 🔧 Optimization Methods

    - **Simple LOBF**: Line of Best Fit through ideal pile tops
    - **Refined LOBF**: Vertical shift to minimize grading
    - **Dynamic Fixed**: Optimal fixed-end pile configuration
    - **N-S Constrained**: Adjacent row constraints applied
    """)

# Footer
st.markdown("---")
st.markdown(f"*Solar Pile Optimization Analysis Tool - Version {version}*")
