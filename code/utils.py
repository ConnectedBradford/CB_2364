import re
import os
import pandas as pd
import plotly.express as px
from IPython.display import display, HTML
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import math
import warnings
from rich import print

def print_unique(df, label):
    print(f"{label}: {df['person_id'].nunique()} unique children")

def display_asq_dist(df, text_col="ASQ_CTV3Text", code_col="ASQ_CTV3Code", title="Distribution of ASQ records (ASQ-3, ASQ-SE, Others)"):
    """
    Display categorized distributions of ASQ records (ASQ-3, ASQ-SE, Others) 
    with validation that ensures no record count mismatch.
    
    Parameters
    ----------
    df : pd.DataFrame
        The input DataFrame containing ASQ-related records.
    text_col : str, default='ASQ_CTV3Text'
        Column name containing ASQ description text.
    code_col : str, default='ASQ_CTV3Code'
        Column name containing ASQ code.
    """
    
    df = df.copy()

    # --- Helper: classify ASQ record type ---
    def classify_asq(text):
        if pd.isna(text):
            return "Missing"
        t = text.lower()
        if t.startswith(("asq-se", "asq:se")):
            return "ASQ-SE"
        elif t.startswith(("asq-3", "asq third")):
            match = re.search(r'(\d+)\s*month', t)
            if match:
                return f"ASQ-3 {int(match.group(1))}m"
            else:
                return "ASQ-3 no month"
        else:
            return "Others"

    # --- Value counts summary ---
    vc_df = (
        df.groupby([code_col, text_col])
          .size()
          .reset_index(name="Count")
          .sort_values("Count", ascending=False)
    )
    vc_df["Category"] = vc_df[text_col].apply(classify_asq)
    df["ASQ_Version"] = df[text_col].apply(classify_asq)

    tables = []

    # --- ASQ-SE ---
    tbl = vc_df[vc_df["Category"] == "ASQ-SE"][[code_col, text_col, "Count"]]
    if not tbl.empty:
        tables.append(tbl.style.hide(axis="index").set_caption("Distribution of ASQ-SE records"))

    # --- ASQ-3 by month (sorted by numeric month) ---
    asq3_months = []
    for c in vc_df["Category"].unique():
        match = re.search(r'(\d+)m', c)
        if match:
            asq3_months.append((int(match.group(1)), c)) # (month_number, category)
    asq3_months = [c for _, c in sorted(asq3_months)] # sorted by month

    for cat in asq3_months:
        g = vc_df[vc_df["Category"] == cat]
        tables.append(
            g[[code_col, text_col, "Count"]]
            .reset_index(drop=True)
            .style.hide(axis="index")
            .set_caption(f"Distribution of {cat} records")
        )

    # --- ASQ-3 without month ---
    tbl = vc_df[vc_df["Category"] == "ASQ-3 no month"][[code_col, text_col, "Count"]]
    if not tbl.empty:
        tables.append(tbl.style.hide(axis="index").set_caption("Distribution of ASQ-3 records without month specified"))

    # --- Others ---
    tbl = vc_df[vc_df["Category"] == "Others"][[code_col, text_col, "Count"]]
    if not tbl.empty:
        tables.append(tbl.style.hide(axis="index").set_caption("Distribution of other ASQ records - will be removed"))

    # --- Build collapsible HTML block ---
    html = "<div style='display: flex; flex-wrap: wrap; gap: 20px;'>"
    for styler in tables:
        html += f"<div style='flex: 1;'>{styler.to_html()}</div>"
    html += "</div>"

    wrapped_html = f"""
    <details>
      <summary style="background-color:#f0f8ff; padding:8px; border:1px solid #ccc; border-radius:5px; cursor:pointer; font-weight:bold; color:#003366;">
        Click to expand: {title}
      </summary>
      {html}
    </details>
    """

    # --- Display tables in Jupyter ---
    display(HTML(wrapped_html))

    # --- Validation: ensure all counts add up ---
    total_from_value_counts = vc_df["Count"].sum()
    total_from_split = sum(styler.data["Count"].sum() for styler in tables)

    assert total_from_value_counts == total_from_split, (
        f"Mismatch in counts! value_counts total = {total_from_value_counts}, "
        f"split tables total = {total_from_split}"
    )

    # print(
    #     "Validation passed: When we add up the numbers from all the tables we created, "
    #     "the total is exactly the same as the original count from the raw data. "
    #     "So no records have been lost or double-counted."
    # )

    # --- Return classified df for further use ---
    return df

def count_records_and_person(df, person_col='person_id'):
    return df.shape[0], df[person_col].nunique()

def report_value_counts(df, col_names, id_col='person_id', mode='html', caption='Value Counts', sort_by=None, ascending=False, max_height=420):

    # counts = series.value_counts(dropna=False).reset_index()
    counts = df.groupby(col_names, dropna=False)[id_col].nunique().reset_index()
    
    if isinstance(col_names, str):
        col_names = [col_names]
    
    # counts.columns = col_names + ['count']    
    counts.columns = col_names + ['person_count']    
    
    total_people = counts['person_count'].sum()
    counts['%'] = (counts['person_count'] / total_people * 100).apply(lambda x: "{:.2f}".format(x))
    
    if sort_by is not None:
        counts = counts.sort_values(by=sort_by, ascending=ascending)
    else:
        counts = counts.sort_values(by=counts.columns[0], ascending=ascending)

    # display(HTML(counts.to_html(index=False)))
    
    html_table = counts.to_html(index=False, escape=False)
    
    # styled_html = f"""
    # <div style="display: flex; justify-content: center; margin: 1em 0;">
    #     <div style="border: 1px solid #ccc; border-radius: 5px; padding: 0.5em 1em; 
    #                 background-color: #fafafa; width: auto;">
    #         {html_table}
    #     </div>
    # </div>
    # """
    
    styled_html = f"""
    <div style="display: flex; justify-content: left; margin: 1em 0;">
        <div style="border: 1px solid #ccc; border-radius: 5px; padding: 0.5em 1em; 
                    width: auto; min-width: calc(100% / 3); max_height: {max_height}px; 
                    overflow-y: auto;">
            {html_table}
        </div>
    </div>
    """
    
    if mode == 'html':
        display(HTML(styled_html))
    elif mode == 'df':
        return counts
    elif mode == 'style':
        styled = (
            counts
            .style
            .hide(axis="index")
            .set_caption(caption)
            .format({"count": "{:,}"})
        )
        display(styled)
        
        # return styled
    else:
        raise ValueError(f"Unsupported mode: {mode}. Choose from 'html', 'df', or 'style'.")

def show_person_coverage(
    df: pd.DataFrame,
    person_col: str = "person_id",
    cols: list[str] | None = None,
    max_height: int = 420,
    sort_by: str = "missing_percentage",
    ascending: bool = False,
    return_df: bool = False,
    show_html: bool = True,
):
    """
    Display a scrollable HTML summary of per-column coverage at the *person* level.
    For each column, we compute:
      - person_total: total unique person (by `person_col`)
      - person_with_value: # of unique person who have at least one non-null value in this column
      - person_missing: # of unique person who are missing this column entirely (all rows null)
      - missing_percentage: person_missing / person_total * 100

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to summarize. May contain multiple rows per person.
    person_col : str, default "person_id"
        Column name that identifies a person.
    cols : list[str] | None
        Subset of columns to summarize. If None, summarize all columns except `person_col`.
    max_height : int, default 420
        Height of the scrollable HTML box in pixels.
    sort_by : {"missing_percentage","person_missing","person_with_value", ...}, default "missing_percentage"
        Column to sort the summary by.
    ascending : bool, default False
        Sort order.
    return_df : bool, default False
        If True, return the summary DataFrame (in addition to HTML display).

    Notes
    -----
    - A person is counted as "with value" for a column if they have *any* non-null in that column across their rows.
    - A person is counted as "missing" for a column if all their rows are null in that column.
    """

    if person_col not in df.columns:
        raise KeyError(f"`person_col` '{person_col}' not found in DataFrame columns.")

    # Determine columns to summarize
    if cols is None:
        cols = [c for c in df.columns if c != person_col]
    else:
        cols = [c for c in cols if c != person_col and c in df.columns]

    # Total unique person (denominator)
    person_total = df[person_col].nunique(dropna=True)

    if person_total == 0:
        raise ValueError("No person found (unique `person_id` is 0).")

    # Aggregate once: for each person, does this column have any non-null value?
    # Result shape: (n_person, n_columns). Values are booleans.
    
    # # a bit slow
    # any_non_null_by_person = (
    #     df[cols]
    #     .groupby(df[person_col], sort=False)
    #     .agg(lambda s: s.notna().any())
    # )

    # # For each column, count how many person have at least one value
    # person_with_value = any_non_null_by_person.sum(axis=0).astype(int)
    
    counts_by_person = df.groupby(person_col, sort=False)[cols].count()
    person_with_value = (counts_by_person > 0).sum().astype(int)

    # Derive missing counts and percentages by person
    person_missing = person_total - person_with_value
    missing_percentage = (person_missing / person_total * 100).round(2)

    summary = pd.DataFrame({
        "person_total": person_total,                
        "person_with_value": person_with_value,
        "person_missing": person_missing,
        "missing_percentage": missing_percentage,
    }).loc[cols]  # ensure original column order (except person_col)

    # Sort if requested
    if sort_by in summary.columns:
        summary = summary.sort_values(by=sort_by, ascending=ascending)
    else:
        # Fallback: sort by index if an invalid sort_by is passed
        summary = summary.sort_index()

    # Render as scrollable HTML to avoid notebook truncation
    html_table = summary.to_html(max_rows=None)
    
    # display(HTML(f"""
    # <div style="max_height:{max_height}px; overflow:auto; border:1px solid lightgrey; padding:8px;">
    #   {html_table}
    # </div>
    # """))
    
    styled_html = f"""
    <div style="
        max_height: {max_height}px;
        overflow: auto;
        border: 1px solid lightgrey;
        padding: 8px;
        display: inline-block;
        min-width: calc(100% / 3);
        font-family: monospace;
        text-align: left;
    ">
      <table style="
          border-collapse: collapse;
          width: auto;
      ">
        {html_table}
      </table>
    </div>
    """

    if show_html:
        display(HTML(styled_html))

    if return_df:
        return summary

# def report_changes(before_count, after_count, before_person, after_person, caption="removing duplicates"):
#     print(f"\n📄 {caption.upper()}")
#     print(f"{'Type':<20} {'Before':>12} {'After':>12} {'Change':>10}")
#     print("-" * 60)
#     print(f"{'Total records':<20} {before_count:>12,} {after_count:>12,} {before_count - after_count:>+10,}")
#     print(f"{'Unique person':<20} {before_person:>12,} {after_person:>12,} {before_person - after_person:>+10,}")

def summarise_counts_diff(before_count, after_count, before_person, after_person):
    delta_records = after_count - before_count
    delta_person = after_person - before_person

    df = pd.DataFrame([
        {"Type": "Total records", "Before": before_count, "After": after_count, "Change": delta_records},
        {"Type": "Unique person", "Before": before_person, "After": after_person, "Change": delta_person},
    ])
    return df

def report_changes(
    before_count, after_count, before_person, after_person, 
    caption="removing duplicates", 
    mode="html"
):
    """
    Display or return change summary:
    - mode="html": Render as styled HTML box
    - mode="style": Return DataFrame.style with caption and color
    - mode="df": Return raw DataFrame
    """
    df = summarise_counts_diff(before_count, after_count, before_person, after_person)

    if mode == "df":
        return df

    elif mode == "style":
        styled = (
            df.style
            .hide(axis="index")
            .set_caption(f"{caption.title()}")
            .format({col: "{:,}" for col in ["Before", "After", "Change"]})
            .map(
                lambda v: "color: green" if isinstance(v, (int, float)) and v < 0 else
                          "color: red" if isinstance(v, (int, float)) and v > 0 else "",
                subset=["Change"]
            )
        )
        display(styled)
        # return styled

    elif mode == "html":
        html = f"""
        <div style="margin: 1em 0;">
          <div style="
              display: inline-block;
              font-family: monospace;
              border: 1px solid #ccc;
              border-radius: 5px;
              padding: 0.5em 1em;
              min-width: calc(100% / 3);
              box-sizing: border-box;
          ">
            <div style="font-weight: bold; margin-bottom: 6px;">Change Report: {caption}</div>
            <table style="border-collapse: collapse; width: 100%; table-layout: fixed;">
              <thead>
                <tr>
                  <th style="text-align: left; padding: 8px 12px;">Type</th>
                  <th style="text-align: right; padding: 8px 12px;">Before</th>
                  <th style="text-align: right; padding: 8px 12px;">After</th>
                  <th style="text-align: right; padding: 8px 12px;">Change</th>
                </tr>
              </thead>
              <tbody>
                <tr>
                  <td style="padding: 8px 12px;">Total records</td>
                  <td style="text-align: right; padding: 8px 12px;">{before_count:,}</td>
                  <td style="text-align: right; padding: 8px 12px;">{after_count:,}</td>
                  <td style="text-align: right; padding: 8px 12px; color:{'green' if (after_count - before_count) < 0 else 'red'}">
                      {after_count - before_count:+,}
                  </td>
                </tr>
                <tr>
                  <td style="padding: 8px 12px;">Unique person</td>
                  <td style="text-align: right; padding: 8px 12px;">{before_person:,}</td>
                  <td style="text-align: right; padding: 8px 12px;">{after_person:,}</td>
                  <td style="text-align: right; padding: 8px 12px; color:{'green' if (after_person - before_person) < 0 else 'red'}">
                      {after_person - before_person:+,}
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
        """
        display(HTML(html))

    else:
        raise ValueError(f"Unsupported mode: {mode}. Choose from 'html', 'style', or 'df'.")

def plot_bar(
    df,
    x_col: str,
    y_col: str,
    title: str,
    x_label: str,
    y_label: str,
    text_col: str = None,
    color_col: str = None,
    color_discrete_sequence: list = None,
    show_proportion: bool = False,
    barmode: str = "group",
    total_annotation: bool = True,
    warning_text: str = None,
):
    """
    Create bar plot with optional grouping, text labels, total annotations per group,
    and warning message.

    Parameters
    ----------
    df : DataFrame
        Must contain at least [x_col, y_col].
    x_col : str
        Column name for x-axis categories.
    y_col : str
        Column name for bar heights.
    title : str
        Title of the plot.
    x_label : str
        Label for the x-axis.
    y_label : str
        Label for the y-axis.
    text_col : str, optional
        Column to display as text labels above bars.
    color_col : str, optional
        Column to group bars by.
    show_proportion: bool, optional
        If True, show proportions as part of text labels above bars
    barmode : str, optional
        "group" or "stack" (default = "group").
    total_annotation : bool, optional
        If True, show total annotation(s) on the top-right of the plot.
        - If color_col is None: one total for the whole dataset.
        - If color_col is given: one total per group (each Strategy).
    warning_text : str, optional
        Text to display as a warning annotation.
    """

    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        color=color_col,
        barmode=barmode if color_col else None,
        text=df[text_col] if text_col else None,
        title=title,
        labels={x_col: x_label, y_col: y_label},
        color_discrete_sequence=color_discrete_sequence,
    )

    if text_col:
        if show_proportion:
            fig.update_traces(
            texttemplate="%{text} children<br>%{y:.1%}", 
            textposition="outside"
        )  
        else:
            fig.update_traces(
                texttemplate="%{text}" if text_col else None,
                textposition="outside"
            )

    fig.update_yaxes(
        range=[0, df[y_col].max() * 1.4],
        tickfont=dict(size=12),
        tickformat=".0%" if df[y_col].max() <= 1 else None
    )
    fig.update_layout(
        xaxis=dict(title_font=dict(size=12)),
        title_font=dict(size=16),
    )

    # Add totals
    if total_annotation:
        if color_col:
            # one total per group
            totals = df.groupby(color_col)[text_col].sum().to_dict()
            text_lines = [f"{k}: {v:,}" for k, v in totals.items()]
            total_text = " | ".join(text_lines)
        else:
            # one overall total
            total_val = df[text_col].sum() if text_col else df[y_col].sum()
            total_text = f"Total: {total_val:,} children"

        fig.add_annotation(
            x=1.0, y=1.0,
            xref="paper", yref="paper",
            text=total_text,
            showarrow=False,
            font=dict(size=14, color="black"),
            align="right",
            bgcolor="white",
            bordercolor="black",
            borderwidth=1,
            borderpad=6
        )

    if warning_text:
        fig.add_annotation(
            x=0.0, y=1.0,
            xref="paper", yref="paper",
            text=f"⚠️ {warning_text}",
            showarrow=False,
            font=dict(size=12, color="red"),
            align="center",
            bgcolor="white",
            bordercolor="red",
            borderwidth=1,
            borderpad=6
        )
        
        fig.update_yaxes(
            range=[0, df[y_col].max() * 1.5],
        )
        
    fig.show()
    
def compute_hv_dist(df, mode="latest", cutoff_30m=False):
    """
    Compute home visit distribution of children by age in months.

    Parameters
    ----------
    df : DataFrame
        Must contain HV_CTV3Code, HV_DateEvent, birth_datetime, person_id.
    mode : str
        "latest" = keep latest visit
        "first" = keep first visit
    cutoff_30m : bool
        If True, only keep visits <= 30 months, then take latest.
    """
    
    df = df.copy()
    
    if cutoff_30m:
        df["cutoff_2_5"] = df["birth_datetime"] + pd.DateOffset(days=914)
        df = df[df["HV_DateEvent"] <= df["cutoff_2_5"]].copy()
        df = df.sort_values("HV_DateEvent").groupby("person_id").tail(1)

    else:
        if mode == "latest":
            df = (
                df.sort_values("HV_DateEvent")
                .drop_duplicates(subset="person_id", keep="last")
            )
        elif mode == "first":
            df = (
                df.sort_values("HV_DateEvent")
                .drop_duplicates(subset="person_id", keep="first")
            )
        elif mode == "unprocessed":
            # do nothing, keep all visits
            pass

    hv_month_counts = (
        df.groupby("age_months_at_HV")["person_id"]
        .nunique()
        .reset_index(name="num_children")
        .sort_values("age_months_at_HV")
    )

    total_children = df["person_id"].nunique()
    hv_month_counts["proportion"] = hv_month_counts["num_children"] / total_children
    return hv_month_counts, total_children

def plot_butterfly(
    df,
    group_col='Group',
    left_col='Completed',
    right_col='NotCompleted',
    title='Comparison by Group',
    left_label='Completed',
    right_label='Not Completed',
    color_left='#4F6658',
    color_right='#A9BDA8'
):
    """
    Plot a symmetric butterfly bar chart for comparing two paired metrics (e.g. completed vs not completed).

    Args:
        df (pd.DataFrame): Input data with one row per group.
        group_col (str): Column name for group/category labels.
        left_col (str): Column name for the left-side values (will be shown as negative).
        right_col (str): Column name for the right-side values.
        title (str): Chart title.
        left_label (str): Label text shown above the left side.
        right_label (str): Label text shown above the right side.
        color_left (str): Color for the left-side bars.
        color_right (str): Color for the right-side bars.
    """

    # Ensure "All" row is always on top
    group_values = df[group_col].tolist()
    if 'All' in group_values:
        group_values = ['All'] + [g for g in group_values if g != 'All']
    df[group_col] = pd.Categorical(df[group_col], categories=group_values[::-1], ordered=True)

    # Percentages
    df['Total'] = df[left_col] + df[right_col]
    df['Left_pct'] = df[left_col] / df['Total'] * 100
    df['Right_pct'] = df[right_col] / df['Total'] * 100

    def get_font_color(bg_color):
        """Choose white or dark font based on background brightness."""
        rgb = tuple(int(bg_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        luminance = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return '#ffffff' if luminance < 140 else '#333333'

    font_color_left = get_font_color(color_left)
    font_color_right = get_font_color(color_right)

    # X-axis ticks
    max_val = max(df[left_col].max(), df[right_col].max())
    tick_step = max(100, int(max_val / 5))
    tick_vals = list(range(-max_val, max_val + 1, tick_step))
    tick_text = [str(abs(v)) for v in tick_vals]

    fig = go.Figure()

    # ➤ Left side (negative)
    fig.add_trace(go.Bar(
        y=df[group_col],
        x=[-v for v in df[left_col]],
        orientation='h',
        text=[f"<b>{v} ({p:.1f}%)</b>" for v, p in zip(df[left_col], df['Left_pct'])],
        textposition='inside',
        insidetextanchor='middle',
        marker_color=color_left,
        textfont=dict(color=font_color_left, size=12),
        hoverinfo='skip',
    ))

    # ➤ Right side
    fig.add_trace(go.Bar(
        y=df[group_col],
        x=df[right_col],
        orientation='h',
        text=[f"<b>{v} ({p:.1f}%)</b>" for v, p in zip(df[right_col], df['Right_pct'])],
        textposition='inside',
        insidetextanchor='middle',
        marker_color=color_right,
        textfont=dict(color=font_color_right, size=12),
        hoverinfo='skip',
    ))

    # Center line
    fig.add_vline(x=0, line_color='gray', line_width=1)

    # Top labels
    fig.add_annotation(
        x=-df[left_col].max() * 0.5,
        yref='paper',
        y=1.12,
        text=f"<b>{left_label}</b>",
        showarrow=False,
        font=dict(size=15, color=color_left),
    )
    fig.add_annotation(
        x=df[right_col].max() * 0.5,
        yref='paper',
        y=1.12,
        text=f"<b>{right_label}</b>",
        showarrow=False,
        font=dict(size=15, color=color_right),
    )

    # Layout
    fig.update_layout(
        barmode='overlay',
        title=dict(text=f'<b>{title}</b>', x=0.5),
        xaxis=dict(
            title='<b>Number of children</b>',
            showgrid=False,
            zeroline=True,
            zerolinecolor='gray',
            zerolinewidth=1,
            tickvals=tick_vals,
            ticktext=tick_text,
            tickfont=dict(size=12),
            linecolor='black',
        ),
        yaxis=dict(
            title='',
            categoryorder='array',
            categoryarray=group_values[::-1],
            tickfont=dict(size=13),
        ),
        plot_bgcolor='white',
        showlegend=False,
        margin=dict(l=50, r=50, t=80, b=40),
    )

    fig.show()

def plot_butterfly_centered(
    df: pd.DataFrame,
    group_col: str,
    left_col: str,
    right_col: str,
    group_order=None,                 # e.g., ['Male', 'Female', 'All']; if None, use the order of first appearance
    left_title: str = "Left",
    right_title: str = "Right",
    color_left: str = "#A9BDA8",
    color_right: str = "#4F6658",
    show_inside_labels: str = "both",  # show "both", "count", or "none"  inside bars
    disable_hover: bool = False,
    center_width: float = 0.006,      # narrow center strip (paper width) used only to align labels
    hspace: float = 0.02,             # horizontal spacing between the three subplot columns (paper coords)
    height: int = 540,
    title: str = None
):
    """
    Plot a butterfly bar chart with centered y-axis labels.
    """
    
    # Work on a copy and keep only required columns
    data = df.copy()
    data = data[[group_col, left_col, right_col]].dropna()
    # data = df[[group_col, left_col, right_col]]

    # Category order for y (top to bottom)
    if group_order is None:
        # group_order = list(pd.unique(data[group_col].astype(str)))
        group_order = list(pd.unique(data[group_col]))
    
    # data[group_col] = pd.Categorical(
    #     data[group_col].astype(str),
    #     categories=group_order,
    #     ordered=True
    # )

    # Sort by the categorical order and compute totals/percentages
    data = data.sort_values(group_col)
    data["Total"] = data[left_col].astype(float) + data[right_col].astype(float)
    data["pct_left"]  = np.where(data["Total"] > 0, data[left_col]  / data["Total"] * 100, 0.0)
    data["pct_right"] = np.where(data["Total"] > 0, data[right_col] / data["Total"] * 100, 0.0)

    # Utilities
    def get_font_color(bg_hex: str) -> str:
        """Choose black/white text for sufficient contrast against the given hex background color."""
        rgb = tuple(int(bg_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        luminance = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
        return '#ffffff' if luminance < 140 else '#333333'

    def fmt_int(x: float) -> str:
        """Format integers with thin spaces for thousands (easier to read)."""
        try:
            return f"{int(round(float(x))):,}".replace(",", " ")
        except Exception:
            return str(x)

    font_color_left  = get_font_color(color_left)
    font_color_right = get_font_color(color_right)

    # Data-driven tick construction (nice ticks that reach/cover the max)
    def auto_ticks(values, n_target=5):
        """
        Create a clean set of ticks based on real data scale.
        Always starts at 0 and extends to a rounded-up maximum.
        """
        vmin, vmax = 0, float(np.nanmax(values)) if len(values) else 0.0
        if vmax <= 0:
            return [0]
        raw_step = vmax / max(n_target, 1)
        step = 10 ** math.floor(math.log10(raw_step))
        for mult in [1, 2, 2.5, 5, 10]:
            if step * mult >= raw_step:
                step = step * mult
                break
        tick_max = math.ceil(vmax / step) * step
        ticks = list(np.arange(0, tick_max + 0.5 * step, step))
        return ticks

    tick_vals_left = auto_ticks(data[[left_col]].values.flatten())
    tick_vals_right = auto_ticks(data[[right_col]].values.flatten())
    tick_text_left = [fmt_int(v) for v in tick_vals_left]
    tick_text_right = [fmt_int(v) for v in tick_vals_right]

    # Three-column layout: left | very narrow center | right
    left_w = right_w = (1.0 - center_width) / 2.0
    column_widths = [left_w, center_width, right_w]
    
    # determine bar text
    if show_inside_labels in ("none", False):
        bar_text_left = None
        bar_text_right = None
    elif show_inside_labels == "count":
        bar_text_left = [f"<b>{fmt_int(v)}</b>" for v in data[left_col]]
        bar_text_right = [f"<b>{fmt_int(v)}</b>" for v in data[right_col]]
    else:   # "both"
        bar_text_left = [
            f"<b>{fmt_int(v)} ({p:.1f}%)</b>"
            for v, p in zip(data[left_col], data["pct_left"])
        ]
        bar_text_right = [
            f"<b>{fmt_int(v)} ({p:.1f}%)</b>"
            for v, p in zip(data[right_col], data["pct_right"])
        ]

    # choose hover behavior
    if disable_hover:
        hover_left = {"hoverinfo": "skip"}
        hover_right = {"hoverinfo": "skip"}
    else:
        hover_left = {"hovertemplate": f"{group_col}=%{{y}}<br>{left_col}=%{{x}}<extra></extra>"}
        hover_right = {"hovertemplate": f"{group_col}=%{{y}}<br>{right_col}=%{{x}}<extra></extra>"}


    fig = make_subplots(
        rows=1, cols=3,
        shared_yaxes=True,
        column_widths=column_widths,
        horizontal_spacing=hspace
    )

    # Left panel (mirrored to the left by reversing x-axis)
    fig.add_trace(go.Bar(
        y=data[group_col],
        x=data[left_col],
        orientation='h',
        text=bar_text_left,
        textposition='inside',
        insidetextanchor='middle',
        marker_color=color_left,
        textfont=dict(color=font_color_left, size=12),
        name=left_title,
        **hover_left
    ), row=1, col=1)

    # Right panel
    fig.add_trace(go.Bar(
        y=data[group_col],
        x=data[right_col],
        orientation='h',
        text=bar_text_right,
        textposition='inside',
        insidetextanchor='middle',
        marker_color=color_right,
        textfont=dict(color=font_color_right, size=12),
        name=right_title,
        **hover_right
    ), row=1, col=3)

    # Axes: remove bottom x-axis titles; we will add two top section headers instead
    fig.update_xaxes(
        autorange='reversed',
        tickvals=tick_vals_left, ticktext=tick_text_left,
        showgrid=False, showticklabels=True,
        linecolor='black', title_text=None,
        row=1, col=1
    )
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_xaxes(
        tickvals=tick_vals_right, ticktext=tick_text_right,
        showgrid=False, showticklabels=True,
        linecolor='black', title_text=None,
        row=1, col=3
    )

    # Hide all y-axis ticks/labels so that only the center annotations act as "y labels"
    for c in (1, 2, 3):
        fig.update_yaxes(
            categoryorder='array', categoryarray=group_order,
            showline=False, showgrid=False,
            ticks="", showticklabels=False,
            row=1, col=c
        )

    # Compute the paper x-coordinate of the center strip to place group labels
    total_space = 2 * hspace         # two gaps between the three columns
    scale = (1.0 - total_space)      # drawable width after subtracting the gaps
    d1 = left_w * scale              # left domain width (paper coord)
    d2 = center_width * scale        # center domain width (paper coord)
    dom1 = (0.0, d1)                 # left domain start/end (paper coord)
    dom2 = (dom1[1] + hspace, dom1[1] + hspace + d2)
    center_x_paper = (dom2[0] + dom2[1]) / 2.0

    # Add centered group labels as annotations in the center strip
    present_groups = set(data[group_col])
    for g in group_order:
        if g in present_groups:
            fig.add_annotation(
                xref='paper', yref='y',
                x=center_x_paper, y=g,
                text=f"<b>{g}</b>",
                showarrow=False,
                font=dict(size=15, color='black'),
                align='center',
                xanchor='center'
            )

    # Layout and main title
    fig.update_layout(
        title=dict(text=title, x=0.5),
        barmode='overlay',
        plot_bgcolor='white',
        paper_bgcolor='white',
        showlegend=False,  # set True if you want a legend
        margin=dict(l=40, r=40, t=80, b=40),
        height=height
    )

    # Two top-only section headers (no bottom axis titles)
    fig.add_annotation(
        x=dom1[0] + (dom1[1]-dom1[0])/2,   # center of the left panel in paper coords
        xref='paper', yref='paper',
        y=1.10,
        text=f"<b>{left_title}</b>",
        showarrow=False,
        font=dict(size=16, color=color_left)
    )
    fig.add_annotation(
        x=1 - (dom1[1]-dom1[0])/2,         # center of the right panel in paper coords
        xref='paper', yref='paper',
        y=1.10,
        text=f"<b>{right_title}</b>",
        showarrow=False,
        font=dict(size=16, color=color_right)
    )

    return fig

def show_tab_model(file_path, delete_file=False):
    """
    Display an HTML file (e.g., sjPlot::tab_model output) inline in Jupyter.
    Handles numpy arrays, file existence, and encoding issues automatically.
    """
    
    if file_path:

        # --- Convert numpy array to normal string ---
        if isinstance(file_path, (np.ndarray, list)):
            try:
                file_path = file_path[0]
            except Exception:
                file_path = file_path.item()

        # --- Ensure path is now a string ---
        if not isinstance(file_path, str):
            raise TypeError(f"Expected file_path to be a string, got {type(file_path)} instead.")

        # --- Check file existence ---
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"HTML file not found at: {file_path}")

        # --- Read and display HTML ---
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            display(HTML(html_content))
        except UnicodeDecodeError:
            # fallback encoding
            with open(file_path, "r", encoding="latin-1") as f:
                html_content = f.read()
            display(HTML(html_content))   
        
        if delete_file:
            os.remove(file_path)
        
    else:
        print("No models could be fitted due to insufficient variation in predictors.") 

def make_crosstab(
    df,
    row_var,
    col_var,
    id_var="person_id",
    caption_prefix=None,
    pct=True,
    show_total=True,
    dropna=False,
):
    """
    Robust crosstab with:
    - counts
    - percentages (of total)
    - row/column totals
    - no pandas background_gradient bugs
    """
    
    if isinstance(row_var, str):
        row_var = [row_var]
    if isinstance(col_var, str):
        col_var = [col_var]

    count_tab = pd.crosstab(
        index=[df[v] for v in row_var],
        columns=[df[v] for v in col_var],
        dropna=dropna
    )

    total_n = count_tab.to_numpy().sum()
    total_children = df[id_var].nunique()
    # assert total_children == total_n, (
    #     f"Mismatch: unique {id_var}={total_children}, crosstab={total_n}"
    # )
    # warnings.warn(f"Mismatch: unique {id_var}={total_children}, crosstab={total_n}") if total_children != total_n else None
    if total_children != total_n:
        print(f"[bold red]⚠️ Warning: Mismatch: unique {id_var}={total_children}, crosstab={total_n} [/bold red] ")
    
    row_names = count_tab.index.names
    
    if len(col_var) > 1:
        count_tab.columns = [
            ",\n".join([f"{name}={val}" for name, val in zip(count_tab.columns.names, key)])
            if isinstance(key, tuple) else str(key)
            for key in count_tab.columns
        ]
    
    if show_total == True:
        count_tab["Total"] = count_tab.sum(axis=1)
        total_row = count_tab.sum().to_frame().T
        total_row.index = ["Total"]
        count_tab = pd.concat([count_tab, total_row], axis=0)
        count_tab.index.name = row_names[0]
    
    if pct == True:
        pct_tab = (count_tab / total_children * 100).round(2)

        final_tab = count_tab.astype(int).astype(str) + " (" + pct_tab.astype(str) + "%)"
    
    else:
        final_tab = count_tab
    
    final_tab.columns = pd.MultiIndex.from_product([[' and '.join(col_var)], final_tab.columns], names=None)

    # styler = final_tab.style.set_caption(
    #     f"{caption_prefix} (Count & % of total = {total_children})" if pct else f"{caption_prefix} (total = {total_children})"
    # )

    # formatting
    # styler = styler.set_properties(**{
    #     "text-align": "center",
    #     "white-space": "nowrap"
    # })
    styler = (
        final_tab.style
            .set_caption(
                f"{caption_prefix} (Count & % of total = {total_children})" if pct else f"{caption_prefix} (total = {total_children})"
            )
            .set_table_styles(
                [
                    # Column level 0
                    {"selector": "th.col_heading.level0", 
                     "props": [("text-align", "center")]},

                    # Column level 1
                    {"selector": "th.col_heading.level1", 
                     "props": [("text-align", "center")]}
                ]
            )
            .set_properties(**{"text-align": "center"})  # 表格内容居中
    )

    display(styler)
    return final_tab
