'''
This module handles the methods used for pdf processing.

Features:
- Compile PDF for test, followup and team report.

'''

### Imports ###

import logging
from utils.data_processing import *
import pandas as pd
import os
import subprocess
import latexcodec
import random
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt

def clear_folder(folder_path):
    """
    Clear all files in the specified folder, except for .gitkeep.

    This function iterates through the given folder, removes all files and links,
    and leaves the .gitkeep file intact.

    Parameters
    ----------
    folder_path : str
        The path to the folder that should be cleared.

    Returns
    -------
    None
        This function does not return any value.
    """
    try:
        # Check if the folder exists
        if os.path.exists(folder_path):
            for filename in os.listdir(folder_path):
                if filename != ".gitkeep":
                    file_path = os.path.join(folder_path, filename)
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
    
    except Exception as e:
        logging.error(f"Error in clear_folder: {e}\n{traceback.format_exc()}")
        raise

def compile_pdf(string, job):
    """
    Compile a LaTeX string into a PDF file using latexmk.

    This function creates a LaTeX `.tex` file from the provided string, 
    then uses `latexmk` to compile it into a PDF and saves it in a temporary directory.

    Parameters
    ----------
    string : str
        The LaTeX code to be compiled into a PDF.
    
    job : str
        The name for the output PDF file (without the extension).

    Returns
    -------
    None
        The function does not return any value. The PDF is saved in the './temp' directory.
    """
    try:
        # Write the LaTeX string to a .tex file
        tex_file_path = f"./temp/{job}.tex"
        with open(tex_file_path, "w") as f:
            f.write(string)

        # Compile the LaTeX code into a PDF using latexmk
        args = ["latexmk", "-pdf", f"-jobname={job}", "-output-directory=./temp", tex_file_path]
        subprocess.run(args)

    except Exception as e:
        logging.error(f"Error in compile_pdf: {e}\n{traceback.format_exc()}")
        raise

def generate_tabular_data(variable_names, variable_keys, units, report_results, use_lactate):
    """
    Generate tabular data for LaTeX report.

    This function creates LaTeX-formatted tabular strings for variable names, their corresponding values,
    and their units, based on the provided `report_results`.

    Parameters
    ----------
    variable_names : list of str
        The names of the variables to be included in the table.
    variable_keys : list of str
        The keys to access the variable values from `report_results`.
    units : list of str
        The units corresponding to each variable in `variable_names`.
    report_results : dict
        A dictionary containing the results for each variable, accessed by keys from `variable_keys`.
    use_lactate : bool
        Flag to indicate whether to include lactate data in the table. If `False`, lactate data is excluded.

    Returns
    -------
    list of str
        A list of three LaTeX-formatted strings, each corresponding to a row of the table.
    """
    try:
        # Exclude lactate if specified
        if not use_lactate:
            variable_names = variable_names[:-1]
            units = units[:-1]

        # Populate the tabular data
        tab_strings = ["", "", ""]
        for s in range(3): 
            for i, var_name in enumerate(variable_names):
                tab_strings[s] += fr"\textbf{{{var_name}}} & {report_results[variable_keys[i]][s]} {units[i]} \\"

        return tab_strings
    
    except Exception as e:
        logging.error(f"Error in generate_tabular_data: {e}\n{traceback.format_exc()}")
        raise

def process_lactate_dataframe(df_lactate, levels, colors, test_weight, report_results):
    """
    Process lactate data and generate LaTeX-formatted table rows.

    This function processes the lactate data in `df_lactate` by iterating over the rows, applying
    color formatting based on predefined thresholds, and generating a LaTeX-compatible string for 
    the table. The string includes various physiological metrics such as speed, lactate, VO2, 
    glucose ratio, lipids ratio, power, and perceived exertion.

    Parameters
    ----------
    df_lactate : pandas.DataFrame
        DataFrame containing lactate data including columns like 'vitesse', 'pente', 'fc', 'lactate', 
        'puissance', and 'rpe'.
    
    levels : dict
        Dictionary containing the levels information (e.g., 'x' for indices and 'vo2' for VO2 values).
    
    colors : list of str
        List of colors for the table row highlighting at each lactate threshold.
    
    test_weight : float
        Weight of the subject in the test, used to normalize power.
    
    report_results : dict
        Dictionary containing additional report data, like glucose and lipid ratios.
    
    Returns
    -------
    str
        A LaTeX-formatted string containing table rows with data from `df_lactate` and `report_results`.
    """
    try:
        extended_tab = ""
        t = 0
        df_lactate['lactate'] = np.round(df_lactate['lactate'], 1)
        df_lactate = df_lactate.replace(np.nan, 'N/A')

        # Iterate over each row in the lactate DataFrame
        for i, row in df_lactate.iterrows():
            # Highlight row if it corresponds to a lactate threshold
            if t < 2 and i == round(levels['x'][t]):
                extended_tab += fr" \rowcolor{{{colors[t]}}} "
                t += 1
            elif i == len(df_lactate) - 1:
                extended_tab += r" \rowcolor{redmax} "

            # Format each row for LaTeX table
            extended_tab += (
                fr"{i} & {row['vitesse']} & {row['pente']} & {row['fc']} & {row['lactate']} & "
                fr"{round(levels['vo2'][i], 1)} & {report_results['glu_ratio'][i]} & {report_results['lip_ratio'][i]} & "
                fr"{round(row['puissance'], 1)} & {round(row['puissance'] / test_weight, 1)} & {row['rpe']} \\ \hline"
            )
        return extended_tab
    
    except Exception as e:
        logging.error(f"Error in process_lactate_dataframe: {e}\n{traceback.format_exc()}")
        raise

def generate_latex_team(team):
    """
    Generate a LaTeX-formatted team report.

    This function generates a LaTeX report for a team. The report includes sections such as:
    - Team header with logos
    - Maximal values plot
    - Thresholds values plot

    Parameters
    ----------
    team : object
        A team object that should have a `name` attribute. This attribute is used to customize 
        the title of the report.

    Returns
    -------
    str
        A LaTeX-formatted string that can be compiled into a PDF.
    """
    try:
        # Create the LaTeX code with embedded team name
        latex_formatted_team_report = r"""
            \documentclass{article}

            % Packages
            \usepackage{graphicx}
            \usepackage[a4paper, margin=0.5cm]{geometry}
            \usepackage{xcolor}
            \usepackage{tgadventor}
            \renewcommand*\familydefault{\sfdefault}
            \usepackage[T1]{fontenc}
            \usepackage[most]{tcolorbox}
            \usepackage{tabularx}
            \renewcommand{\arraystretch}{1.2}
            \usepackage{colortbl}

            % Color settings
            \definecolor{ballblue}{rgb}{0.13,0.67,0.8}
            \definecolor{lightlightgray}{rgb}{0.93,0.93,0.93}
            \definecolor{greens1}{RGB}{88,214,141}
            \definecolor{oranges2}{RGB}{248,196,113}
            \definecolor{redmax}{RGB}{229,152,102}

            % Delete number of page
            \pagestyle{empty}

            % Document creation
            \begin{document}

                \centering

                % Header 
                \begin{figure}[!h]
                    \centering
                    \begin{minipage}{0.15\linewidth}
                        \centering
                        \includegraphics[width = \linewidth]{assets/cnsnmm.png}
                    \end{minipage}
                    \begin{minipage}{0.65\linewidth}
                        \centering
                        \huge{\textcolor{ballblue}{BILAN D'EQUIPE : """+team.name.upper().encode('latex').decode('utf-8')+r"""}} \\
                        \normalsize{CENTRE NATIONAL DE SKI NORDIQUE \& DE MOYENNE MONTAGNE}
                    \end{minipage}
                    \hspace{0.04\linewidth}
                    \begin{minipage}{0.10\linewidth}
                        \centering
                        \includegraphics[width = \linewidth]{assets/ffs.png}
                    \end{minipage}
                    \hspace{0.04\linewidth}
                \end{figure}

                % Maximal values title
                \begin{tcolorbox}[colback=ballblue,colframe=ballblue,left=0em,right=0em,width=0.45\linewidth]
                    \centering
                    \large
                    \textcolor{white}{DISTRIBUTION DES VALEURS MAXIMALES}
                \end{tcolorbox}

                % Maximal values plots
                \begin{tcolorbox}[colback=lightlightgray,colframe=lightlightgray,left=0em, right=0em, bottom=0em, top=0em]
                    \centering
                    \includegraphics[width=\linewidth]{./temp/temp_team_max.png}
                \end{tcolorbox}

                % Thresholds values title
                \begin{tcolorbox}[colback=ballblue,colframe=ballblue,left=0em,right=0em,width=0.3\linewidth]
                    \centering
                    \large
                    \textcolor{white}{DISTRIBUTION DES SEUILS}
                \end{tcolorbox}

                % Thresholds values plots
                \begin{tcolorbox}[colback=lightlightgray,colframe=lightlightgray,left=0em, right=0em, bottom=0em, top=0em]
                    \centering
                    \includegraphics[width=\linewidth]{./temp/temp_team_threshold.png}
                \end{tcolorbox}
                
            \end{document}
            """
        
        return latex_formatted_team_report
    
    except Exception as e:
        logging.error(f"Error in generate_latex_team: {e}\n{traceback.format_exc()}")
        raise

def generate_latex_followup(athlete, latex_tabular_values, levels_labels, levels_units):
    """
    Generate a LaTeX-formatted follow-up report for an athlete.

    This function generates a LaTeX report for an athlete that includes:
    - Athlete header with name
    - Lactate and VO2 values plots
    - Threshold values plots
    - Tabular data for athlete measurements (e.g., VO2, lactate levels, etc.)

    Parameters
    ----------
    athlete : object
        An athlete object with `first_name` and `last_name` attributes to customize the header.
    latex_tabular_values : str
        A LaTeX-formatted string containing tabular data to be included in the report.
    levels_labels : list of str
        Labels for different training levels (e.g., "Level 1", "Level 2", etc.).
    levels_units : str
        The unit of measurement for the training levels (e.g., "mL/min", "bpm").

    Returns
    -------
    str
        A LaTeX-formatted string that can be compiled into a PDF.
    """
    try:
        # Create the LaTeX code with embedded athlete name and other dynamic content
        latex_formatted_followup_report = r"""
            \documentclass{article}

            % Packages
            \usepackage{graphicx}
            \usepackage[a4paper, margin=0.5cm]{geometry}
            \usepackage{xcolor}
            \usepackage{tgadventor}
            \renewcommand*\familydefault{\sfdefault}
            \usepackage[T1]{fontenc}
            \usepackage[most]{tcolorbox}
            \usepackage{tabularx}
            \renewcommand{\arraystretch}{1.2}
            \usepackage{colortbl}

            % Color settings
            \definecolor{ballblue}{rgb}{0.13,0.67,0.8}
            \definecolor{lightlightgray}{rgb}{0.93,0.93,0.93}
            \definecolor{greens1}{RGB}{88,214,141}
            \definecolor{oranges2}{RGB}{248,196,113}
            \definecolor{redmax}{RGB}{229,152,102}

            % Delete number of page
            \pagestyle{empty}

            % Document creation
            \begin{document}

                \centering

                % Header 
                \begin{figure}[!h]
                    \centering
                    \begin{minipage}{0.15\linewidth}
                        \centering
                        \includegraphics[width = \linewidth]{assets/cnsnmm.png}
                    \end{minipage}
                    \begin{minipage}{0.65\linewidth}
                        \centering
                        \huge{\textcolor{ballblue}{SUIVI DE : """+athlete.first_name.upper().encode('latex').decode('utf-8')+' '+athlete.last_name.upper().encode('latex').decode('utf-8')+r"""}} \\
                        \normalsize{CENTRE NATIONAL DE SKI NORDIQUE \& DE MOYENNE MONTAGNE}
                    \end{minipage}
                    \hspace{0.04\linewidth}
                    \begin{minipage}{0.10\linewidth}
                        \centering
                        \includegraphics[width = \linewidth]{assets/ffs.png}
                    \end{minipage}
                    \hspace{0.04\linewidth}
                \end{figure}

                % Maximal values title
                \begin{tcolorbox}[colback=ballblue,colframe=ballblue,left=0em,right=0em,width=0.45\linewidth]
                    \centering
                    \large
                    \textcolor{white}{SUIVI DES VALEURS DE LACTATE ET DE VO2}
                \end{tcolorbox}

                % Maximal values plots
                \begin{tcolorbox}[colback=lightlightgray,colframe=lightlightgray,left=0em, right=0em, bottom=0em, top=0em]
                    \centering
                    \includegraphics[width=\linewidth]{./temp/temp_followup_curves.png}
                \end{tcolorbox}

                % Thresholds values title
                \begin{tcolorbox}[colback=ballblue,colframe=ballblue,left=0em,right=0em,width=0.3\linewidth]
                    \centering
                    \large
                    \textcolor{white}{SUIVI DES SEUILS}
                \end{tcolorbox}

                % Thresholds values plots
                \begin{tcolorbox}[colback=lightlightgray,colframe=lightlightgray,left=0em, right=0em, bottom=0em, top=0em]
                    \centering
                    \includegraphics[width=\linewidth]{./temp/temp_followup_threshold.png}
                \end{tcolorbox}

                % Data values title
                \begin{tcolorbox}[colback=ballblue,colframe=ballblue,left=0em,right=0em,width=0.3\linewidth]
                    \centering
                    \large
                    \textcolor{white}{SUIVI DES DONNEES}
                \end{tcolorbox}

                \begin{tcolorbox}[colback=lightlightgray,colframe=lightlightgray,left=0em,right=0em,width=\linewidth]
                    \centering
                    \scriptsize
                    \begin{tabularx}{\linewidth}{|*{12}{>{\centering\arraybackslash}X|}}
                        \hline
                        \textbf{Date} & \textbf{Poids} \newline \textit{\tiny kg} & \textbf{\%VO2 S1} & \textbf{\%VO2 S2} & 
                        \textbf{VO2max} \textit{\tiny mL/min} & \textbf{VO2max} \textit{\tiny mL/min/kg} & 
                        \textbf{VEmax} \textit{\tiny L/min} & \textbf{"""+levels_labels[0]+r"""} \textit{\tiny """+levels_units+r"""} &
                        \textbf{"""+levels_labels[1]+r"""} \textit{\tiny """+levels_units+r"""} & \textbf{"""+levels_labels[2]+r"""} \textit{\tiny """+levels_units+r"""} &
                        \textbf{FCmax} \textit{\tiny bpm} & \textbf{Lac Max} \textit{\tiny mmol/L} \\
                        \hline
                        """+latex_tabular_values+r"""
                    \end{tabularx}
                \end{tcolorbox}
                
            \end{document}
            """
        
        return latex_formatted_followup_report
    
    except Exception as e:
        logging.error(f"Error in generate_latex_followup: {e}\n{traceback.format_exc()}")
        raise

def generate_latex_report(test, athlete, tab_strings, extended_tab, report_results, extended=True):
    """
    Generate a LaTeX report for a physiological test conducted by an athlete.

    This function generates a LaTeX report with the athlete's profile, test results,
    thresholds, and graphical representations. If the `extended` parameter is set to
    True, additional detailed information is included in the report.

    Parameters
    ----------
    test : Test
        The test object containing information about the test (date, remarks, etc.).
    athlete : Athlete
        The athlete object containing personal information (name, height, weight, etc.).
    tab_strings : list of str
        A list of strings representing the tabular results for the test thresholds.
    extended_tab : str
        The detailed data to be included in the extended report.
    report_results : dict
        A dictionary containing the test results (e.g., slopes, speeds, etc.).
    extended : bool, optional
        If True, generates an extended detailed report. Default is True.

    Returns
    -------
    str
        The generated LaTeX report as a string.
    """
    try:
        latex_formatted_report = r"""
            \documentclass{article}

            % Packages
            \usepackage{graphicx}
            \usepackage[a4paper, margin=0.5cm]{geometry}
            \usepackage{xcolor}
            \usepackage{tgadventor}
            \renewcommand*\familydefault{\sfdefault}
            \usepackage[T1]{fontenc}
            \usepackage[most]{tcolorbox}
            \usepackage{tabularx}
            \renewcommand{\arraystretch}{1.2}
            \usepackage{colortbl}

            % Color settings
            \definecolor{ballblue}{rgb}{0.13,0.67,0.8}
            \definecolor{lightlightgray}{rgb}{0.93,0.93,0.93}
            \definecolor{greens1}{RGB}{88,214,141}
            \definecolor{oranges2}{RGB}{248,196,113}
            \definecolor{redmax}{RGB}{229,152,102}

            % Delete number of page
            \pagestyle{empty}

            % Document creation
            \begin{document}

                \centering

                % Header 
                \begin{figure}[!h]
                    \centering
                    \begin{minipage}{0.15\linewidth}
                        \centering
                        \includegraphics[width = \linewidth]{assets/cnsnmm.png}
                    \end{minipage}
                    \begin{minipage}{0.65\linewidth}
                        \centering
                        \huge{\textcolor{ballblue}{RAPPORT DE TEST PHYSIOLOGIQUE}} \\
                        \normalsize{CENTRE NATIONAL DE SKI NORDIQUE \& DE MOYENNE MONTAGNE}
                    \end{minipage}
                    \hspace{0.04\linewidth}
                    \begin{minipage}{0.10\linewidth}
                        \centering
                        \includegraphics[width = \linewidth]{assets/ffs.png}
                    \end{minipage}
                    \hspace{0.04\linewidth}
                \end{figure}

                % Profil title
                \begin{tcolorbox}[colback=ballblue,colframe=ballblue,left=0em,right=0em,width=0.3\linewidth]
                    \centering
                    \large
                    \textcolor{white}{PROFIL \& REMARQUES}
                \end{tcolorbox}

                % Profil details
                \noindent
                \begin{tcolorbox}[colback=lightlightgray,colframe=lightlightgray,left=0em,right=0em]
                    \noindent
                    \begin{tabularx}{\linewidth}{*{4}{X}}
                        \textbf{\normalsize Nom} & \textbf{\normalsize Taille} & \textbf{\normalsize Sexe} & \textbf{\normalsize Date de naissance} \\
                        """+athlete.last_name.title().encode('latex').decode('utf-8')+r""" & """+str(athlete.height)+r"""cm & """+athlete.gender+r""" & """+athlete.date_of_birth.strftime("%d/%m/%Y")+r""" \\
                    \end{tabularx}
                    
                    \vspace{0.2em}

                    \noindent
                    \begin{tabularx}{\linewidth}{*{4}{X}}
                        \textbf{\normalsize Pr\'enom} & \textbf{\normalsize Poids} & \textbf{\normalsize Sport} & \textbf{\normalsize Date du test} \\
                        """+athlete.first_name.title().encode('latex').decode('utf-8')+r""" & """+str(athlete.weight)+r"""kg & """+athlete.sport.title()+r""" & """+test.date.strftime("%d/%m/%Y")+r""" \\
                    \end{tabularx}
                
                    \vspace{0.5em}
                
                    \noindent
                    \begin{tabularx}{\linewidth}{X}
                        \textbf{\normalsize Remarques du testeur : } """+test.remarks.encode('latex').decode('utf-8')+r"""
                    \end{tabularx}
                \end{tcolorbox}

                % Results title
                \begin{tcolorbox}[colback=ballblue,colframe=ballblue,left=0em,right=0em,width=0.4\linewidth]
                    \centering
                    \large
                    \textcolor{white}{RESULTATS DU TEST \& GRAPHIQUES}
                \end{tcolorbox}

                % Results details
                % First threshold
                \noindent
                \begin{minipage}[t]{0.32\linewidth}
                    \begin{tcolorbox}[colback=lightlightgray,colframe=lightlightgray,left=0em,right=0em]
                        \centering
                        \textbf{SEUIL 1 :} \footnotesize{"""+str(report_results["slope"][0])+r""" \% - """+str(report_results["speed"][0])+r""" km/h}
                        
                        \vspace{0.5em}
                        
                        \footnotesize
                        \begin{tabularx}{\linewidth}{{>{\centering\arraybackslash}X|>{\centering\arraybackslash}X}}
                            """+tab_strings[0]+r"""
                        \end{tabularx}
                    \end{tcolorbox}
                \end{minipage}
                \hfill
                % Second threshold
                \begin{minipage}[t]{0.32\linewidth}
                    \begin{tcolorbox}[colback=lightlightgray,colframe=lightlightgray,left=0em,right=0em]
                        \centering
                        \textbf{SEUIL 2 :} \footnotesize{"""+str(report_results["slope"][1])+r""" \% - """+str(report_results["speed"][1])+r""" km/h}
                        
                        \vspace{0.5em}
                        
                        \footnotesize
                        \begin{tabularx}{\linewidth}{{>{\centering\arraybackslash}X|>{\centering\arraybackslash}X}}
                            """+tab_strings[1]+r"""
                        \end{tabularx}
                    \end{tcolorbox}
                \end{minipage}
                \hfill
                % Max values
                \begin{minipage}[t]{0.32\linewidth}
                    \begin{tcolorbox}[colback=lightlightgray,colframe=lightlightgray,left=0em, right=0em]
                        \centering
                        \textbf{Max :} \footnotesize{"""+str(report_results["slope"][2])+r""" \% - """+str(report_results["speed"][2])+r""" km/h - """+str(report_results['tps_last'])+r"""s}
                        
                        \vspace{0.5em}
                        
                        \footnotesize
                        \begin{tabularx}{\linewidth}{{>{\centering\arraybackslash}X|>{\centering\arraybackslash}X}}
                            """+tab_strings[2]+r"""
                        \end{tabularx}
                    \end{tcolorbox}
                \end{minipage}

                % Graphics part
                \begin{tcolorbox}[colback=lightlightgray,colframe=lightlightgray,left=0em, right=0em, bottom=0em, top=0em]
                    \centering
                    \includegraphics[width=\linewidth]{./temp/temp_graphics.png}
                \end{tcolorbox}
            """

        latex_extended_report = r"""
            % Detailled report
            \newpage

            % Protocole title
            \begin{tcolorbox}[colback=ballblue,colframe=ballblue,left=0em,right=0em,width=0.4\linewidth]
                \centering
                \large
                \textcolor{white}{PROTOCOLE}
            \end{tcolorbox}

            % Protocole graph
            \begin{tcolorbox}[colback=lightlightgray,colframe=lightlightgray,left=0em, right=0em]
                \centering
                \includegraphics[width=\linewidth]{./temp/temp_protocol.png}
            \end{tcolorbox}

            % Detailled results title
            \begin{tcolorbox}[colback=ballblue,colframe=ballblue,left=0em,right=0em,width=0.4\linewidth]
                \centering
                \large
                \textcolor{white}{RESULTATS DETAILLES}
            \end{tcolorbox}

            % Excel file copy
            \renewcommand{\arraystretch}{1.8}
            \begin{tcolorbox}[colback=lightlightgray,colframe=lightlightgray,left=0em, right=0em]
                \centering
                \begin{tabularx}{\linewidth}{|*{11}{>{\centering\arraybackslash}X|}}
                    \hline
                    \textbf{Palier} & \textbf{Vitesse} & \textbf{Pente} & \textbf{FC} & \textbf{Lactate} & \textbf{VO2} & \textbf{\%GLU} & \textbf{\%LIP} & \textbf{Watt} & \textbf{Watt/kg} & \textbf{RPE} \\
                    \hline
                    """+extended_tab+r"""
                \end{tabularx}

                \vspace{1em}
                
                \renewcommand{\arraystretch}{1.5}
                \begin{tabularx}{0.2\linewidth}{|X|>{\centering\arraybackslash}X|}
                    \hline
                    \multicolumn{2}{|c|}{\textbf{L\'egende}} \\
                    \hline
                    \cellcolor{greens1} & Seuil 1 \\
                    \hline
                    \cellcolor{oranges2} & Seuil 2 \\
                    \hline
                    \cellcolor{redmax} & Max \\
                    \hline
                \end{tabularx}
            \end{tcolorbox}

            \newpage

            % VO2 Plateau title
            \begin{tcolorbox}[colback=ballblue,colframe=ballblue,left=0em,right=0em,width=0.4\linewidth]
                \centering
                \large
                \textcolor{white}{ETUDE DU PLATEAU DE VO2}
            \end{tcolorbox}

            % VO2 Plateau study
            \begin{tcolorbox}[colback=lightlightgray,colframe=lightlightgray,left=0em, right=0em]
                \centering
                \includegraphics[width=\linewidth]{./temp/temp_plateau.png}
            \end{tcolorbox}
            
        \end{document}
        """

        # Extended report if needed
        if extended :
            return latex_formatted_report + latex_extended_report
        
        return latex_formatted_report + r"\end{document}"
    
    except Exception as e:
        logging.error(f"Error in generate_latex_report: {e}\n{traceback.format_exc()}")
        raise

def save_temp_files(bytes, path):
    """
    Save the given bytes to a file at the specified path.

    This function writes the byte data to a file in binary mode at the location specified by the `path`.

    Parameters
    ----------
    file_bytes : bytes
        The byte data that should be written to the file.
    path : str
        The path where the file should be saved.

    Returns
    -------
    None
        This function does not return any value.
    """
    try:
        # Open the file in write-binary mode and save the bytes
        with open(path, "wb") as f:
            f.write(bytes)
    
    except Exception as e:
        logging.error(f"Error in save_temp_files: {e}\n{traceback.format_exc()}")
        raise

def compute_pdf(df_lactate, test_id, report_results, levels, use_lactate, extended = True) :
    """
    Compute the LaTeX code for a physiological test report and save temporary files for images.

    This function retrieves the test and athlete data from the database, processes the 
    data to generate tabular and graphical representations, saves temporary image files,
    and generates LaTeX code for the report.

    Parameters
    ----------
    df_lactate : pandas.DataFrame
        The DataFrame containing lactate data.
    test_id : int
        The ID of the test in the database.
    report_results : dict
        A dictionary containing the results of the report (e.g., slopes, speeds).
    levels : list
        A list of lactate levels for the analysis.
    use_lactate : bool
        A flag indicating whether lactate data should be used in the report.
    extended : bool, optional
        A flag indicating whether the extended version of the report should be generated. Default is True.

    Returns
    -------
    str
        The LaTeX code for the test report.
    """
    try:
        # Retrieve the test
        session = next(get_db())
        test = get_by_id(session, Test, test_id)

        # Retrieve the athlete
        athlete = get_by_id(session, Athlete, test.athlete_id)

        # Initialize data
        variable_names = ["VO2", "VO2/kg", r"\%VO2max", "VE", "FC", r"\%FCmax", "Watt", "Watt/kg", "DE", "Glucids", "Lipids", "Lactate"]
        variable_keys = ["vo2", "vo2_kg", "vo2_ratio", "ve", "hr", "hr_ratio", "watt", "watt_kg", "de", "glu", "lip", "lactate"]
        units = ["mL/min", "mL/min/kg", r"\%", "L/min", "bpm", r"\%", "W", "W/kg", "kcal/h", "g/min", "g/min", "mmol/L"]
        colors = ["greens1", "oranges2"]

        # Generate LaTeX table strings
        tab_strings = generate_tabular_data(variable_names, variable_keys, units, report_results, use_lactate)
        extended_tab = process_lactate_dataframe(df_lactate, levels, colors, test.weight, report_results)
        
        # Save temporary files for images
        bytes_images = [test.graphics, test.protocol, test.plateau_study]
        paths = ["./temp/temp_graphics.png", "./temp/temp_protocol.png", "./temp/temp_plateau.png"]
        for i in range(3):
            save_temp_files(bytes_images[i], paths[i])

        # Generate LaTeX code
        latex_code = generate_latex_report(test, athlete, tab_strings, extended_tab, report_results, extended)

        return latex_code
    
    except Exception as e:
        logging.error(f"Error in compute_pdf: {e}\n{traceback.format_exc()}")
        raise

def test_slope_report(tests):
    """
    Test the slopes of the given list of tests and classify them into two categories:
    tests with slopes (non-zero) and tests without slopes (zero).

    This function iterates through the provided tests, calculates the sum of the watt values
    for the first, second, and maximum slopes, and classifies each test into two groups: 
    those with non-zero slopes and those with zero slopes.

    Parameters
    ----------
    tests : list
        A list of test objects, where each test object is expected to have attributes 
        `watt_kg_s1`, `watt_kg_s2`, and `watt_kg_max`.

    Returns
    -------
    tuple
        A tuple containing two lists:
        - The first list contains tests with non-zero slopes.
        - The second list contains tests with zero slopes.
    """
    try:
        # Initialize lists to classify tests
        zero_tests = []
        non_zero_tests = []

        # Classify tests based on slope values
        for t in tests :
            temp_sum = sum([t.watt_kg_s1, t.watt_kg_s2, t.watt_kg_max])
            if temp_sum == 0 :
                zero_tests.append(t)
            else :
                non_zero_tests.append(t)

        return non_zero_tests, zero_tests
    
    except Exception as e:
        logging.error(f"Error in test_slope_report: {e}\n{traceback.format_exc()}")
        raise

def create_team_report(test_ids, team_athletes, bool_zero):
    """
    Generate a team report with visualizations of athlete data, including VO2 max, lactate levels, and thresholds.

    This function generates two main visualizations for a team of athletes:
    1. A subplot with the distribution of VO2 max and lactate max values for each athlete.
    2. A subplot with the distribution of lactate and respiratory thresholds, including comparisons of speed and power.

    Parameters
    ----------
    test_ids : list of int
        A list of test IDs corresponding to the tests performed on each athlete.
    team_athletes : list of Athlete
        A list of athlete objects associated with the tests.
    bool_zero : bool
        A flag that determines whether to compare power (False) or speed (True) for threshold distribution.

    Returns
    -------
    tuple
        A tuple containing two figures:
        - The first figure (`fig_max`) shows the distribution of VO2 max and lactate max.
        - The second figure (`fig_threshold`) shows the distribution of lactate and respiratory thresholds.
    """
    try:
        # Get the tests
        session = next(get_db())
        team_tests = [get_by_id(session, Test, t) for t in test_ids]

        # Create the main figure with subplots
        fig_max = make_subplots(
            rows=1, cols=2,
            horizontal_spacing=0.1,
            subplot_titles=(
                'Distribution des valeurs de VO2 maximales',
                'Distribution des valeurs de lactates maximales'
            )
        )

        # Initialize lists for data to plot
        vo2_max_values = [t.vo2_kg_max for t in team_tests]
        lactate_values = [l.lactate_max for l in team_tests]

        # Colorscale
        colorscale = 'Viridis'

        # Plot data for each athlete
        for index, athlete in enumerate(team_athletes) :
            athlete_name = f"{athlete.first_name.title()} {athlete.last_name.title()}"

            # Handling missing data
            no_data = ''
            if pd.isnull(vo2_max_values[index]):
                no_data += '(VO2-Ø) '
            if pd.isnull(lactate_values[index]):
                no_data += '(Lactate-Ø) '
            
            # Generate a random color for the athlete
            random_color = px.colors.sample_colorscale(colorscale, random.random())[0]

            # Add VO2 max scatter plot trace
            fig_max.add_trace(
                go.Scatter(
                    x=[vo2_max_values[index]], 
                    y=[1],
                    mode="markers",
                    name = f"{athlete_name} {no_data}", 
                    marker = dict(color=random_color, size=15, symbol='x')
                ),
                row=1, col=1
            )

            # Add Lactate max scatter plot trace
            fig_max.add_trace(
                go.Scatter(
                    x=[lactate_values[index]], 
                    y=[1],
                    mode="markers",
                    name = f"{athlete_name} {no_data}", 
                    marker = dict(color=random_color, size=15, symbol='x'),
                    showlegend=False
                ),
                row=1, col=2
            )
            
        # Boxplot of the distribution of VO2 max inside the team
        fig_max.add_box(
            x=vo2_max_values,
            y0=0,   
            line_color="#21ABCC", 
            showlegend=False,
            showwhiskers=True,        
            row=1, col=1
        )

        # Boxplot of the distribution of Lactate max inside the team
        fig_max.add_box(
            x=lactate_values,
            y0=0,
            line_color="#21ABCC",
            showlegend=False,
            showwhiskers=True,                 
            row=1, col=2
        )

        # Update global layout
        for i in range(1, 3):
            title = 'VO2 Max (mL/min/kg)' if i == 1 else 'Lactate Max (mmol/L)'
            fig_max.update_xaxes(
                title_text=title, 
                title_font=dict(size=12),
                row=1, col=i,
                tickfont=dict(size=10),
                showgrid=True
            )
            fig_max.update_yaxes(
                range=[-1, 2],
                row=1, col=i,
                showgrid=False,
                visible=False,
                showticklabels=False
            )
        fig_max.update_layout(
            legend=dict(
                font=dict(size=12),
                xanchor="left",
                yanchor="middle",
                x=1.05,
                y=0.5,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="LightGray",
                borderwidth=1,
            ),
            margin=dict(l=30, r=60, t=30, b=50),
            template="plotly_white",
            height=350
        )

        # Create the thresholds figure and set the parameters
        fig_threshold = make_subplots(
            rows=2, cols=1,
            vertical_spacing=0.12,
            subplot_titles=(
                'Distribution des seuils lactates',
                'Distribution des seuils respiratoires'
            )
        )
        colors_thresholds = ['#21ABCC', '#58D68D', '#D35400']

        # Plot the different thresholds for every athlete
        for index, athlete in enumerate(team_athletes) :
            test = team_tests[index]
            athlete_name = f"{athlete.first_name.title()} {athlete.last_name.title()}"

            # Speed comparison if null power values
            if bool_zero :
                x_label_threshold = 'Vitesse (km/h)'
                fig_threshold.add_trace(
                    go.Scatter(
                        x=[test.speed_s1], 
                        y=[athlete_name],
                        mode="markers",
                        name = f"Seuil 1", 
                        showlegend = True if index == 0 else False,
                        marker = dict(color=colors_thresholds[0], size=15, symbol='x')
                    ),
                    row=1, col=1
                )
                fig_threshold.add_trace(
                    go.Scatter(
                        x=[test.speed_s2], 
                        y=[athlete_name],
                        mode="markers",
                        name = f"Seuil 2", 
                        showlegend = True if index == 0 else False,
                        marker = dict(color=colors_thresholds[1], size=15, symbol='x')
                    ),
                    row=1, col=1
                )
                fig_threshold.add_trace(
                    go.Scatter(
                        x=[test.speed_max], 
                        y=[athlete_name],
                        mode="markers",
                        name = f"Max", 
                        showlegend = True if index == 0 else False,
                        marker = dict(color=colors_thresholds[2], size=15, symbol='x')
                    ),
                    row=1, col=1
                )

            # Power comparison
            else :
                x_label_threshold = 'Puissance (Watt)'
                fig_threshold.add_trace(
                    go.Scatter(
                        x=[test.watt_kg_s1], 
                        y=[athlete_name],
                        mode="markers",
                        name = f"Seuil 1", 
                        showlegend = True if index == 0 else False,
                        marker = dict(color=colors_thresholds[0], size=15, symbol='x')
                    ),
                    row=1, col=1
                )
                fig_threshold.add_trace(
                    go.Scatter(
                        x=[test.watt_kg_s2], 
                        y=[athlete_name],
                        mode="markers",
                        name = f"Seuil 2", 
                        showlegend = True if index == 0 else False,
                        marker = dict(color=colors_thresholds[1], size=15, symbol='x')
                    ),
                    row=1, col=1
                )
                fig_threshold.add_trace(
                    go.Scatter(
                        x=[test.watt_kg_max], 
                        y=[athlete_name],
                        mode="markers",
                        name = f"Max", 
                        showlegend = True if index == 0 else False,
                        marker = dict(color=colors_thresholds[2], size=15, symbol='x')
                    ),
                    row=1, col=1
                )

        # Plot the different %VO2 thresholds for every athlete
        for index, athlete in enumerate(team_athletes) :
            test = team_tests[index]
            athlete_name = f"{athlete.first_name.title()} {athlete.last_name.title()}"
            fig_threshold.add_trace(
                go.Scatter(
                    x=[test.vo2_ratio_s1], 
                    y=[athlete_name],
                    mode="markers",
                    name = f"Seuil 1", 
                    showlegend = False,
                    marker = dict(color=colors_thresholds[0], size=15, symbol='x')
                ),
                row=2, col=1
            )
            fig_threshold.add_trace(
                go.Scatter(
                    x=[test.vo2_ratio_s2], 
                    y=[athlete_name],
                    mode="markers",
                    name = f"Seuil 2", 
                    showlegend = False,
                    marker = dict(color=colors_thresholds[1], size=15, symbol='x')
                ),
                row=2, col=1
            )

        # Update global layout
        for i in range(1, 3):
            title = x_label_threshold if i == 1 else 'VO2/VO2MAX (%)'
            fig_threshold.update_xaxes(
                title_text=x_label_threshold, 
                title_font=dict(size=12),
                row=i, col=1,
                tickfont=dict(size=10),
                showgrid=True
            )
            fig_threshold.update_yaxes(
                title_text='Athlètes',
                title_font=dict(size=12),
                row=i, col=1,
                tickfont=dict(size=10),
                showgrid=True,
            )
        fig_threshold.update_layout(
            legend=dict(
                font=dict(size=12),
                xanchor="left",
                yanchor="middle",
                x=1.05,
                y=0.5,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="LightGray",
                borderwidth=1,
            ),
            margin=dict(l=30, r=70, t=30, b=50),
            template="plotly_white",
            height=800
        )

        return fig_max, fig_threshold
    
    except Exception as e:
        logging.error(f"Error in create_team_report: {e}\n{traceback.format_exc()}")
        raise

def create_followup_report(chosen_tests, bool_zero):
    """
    Generates a follow-up report with plots and tabular data based on the provided tests.
    
    This function generates two main visualizations and a LaTeX table:
    1. A subplot showing lactate and VO2 evolution based on the provided tests.
    2. A subplot showing the distribution of lactate and VO2 max values.
    
    Parameters
    ----------
    chosen_tests : list of Test
        A list of test objects that contain the test data.
    bool_zero : bool
        A flag indicating whether to use speed (True) or power (False) for comparison.
        
    Returns
    -------
    tuple
        A tuple containing:
        - fig_values (plotly.graph_objects.Figure): A figure with plots for lactate and VO2 evolution.
        - fig_thresholds (plotly.graph_objects.Figure): A figure with plots for thresholds.
        - latex_tabular_values (str): LaTeX formatted table of test results.
        - levels_labels (list): Labels for the threshold levels.
        - levels_units (str): Units for the thresholds.
    """
    try:
        # Initialize data containers
        lactate_curves = {}
        vo2_curves = {}
        power_values = None
        tabular_values = []

        # Speed values if null power values
        if bool_zero :

            # Get the lactate, VO2, speed associated threshold and tabular data
            lactate_curves = {t.date : [pd.read_json(StringIO(t.source_excel))['vitesse'], pd.read_json(StringIO(t.source_excel))['lactate']] for t in chosen_tests}
            vo2_curves = {t.date : [pd.read_json(StringIO(t.source_excel))['vitesse'], pd.read_json(StringIO(t.source_excel))['vo2']] for t in chosen_tests}
            power_values = pd.DataFrame([[t.date, t.speed_s1, t.speed_s2, t.speed_max] for t in chosen_tests], columns = ["date", "s1", "s2", "max"])
            tabular_values = [[t.date.strftime("%d-%m-%Y"), str(t.weight), str(t.vo2_ratio_s1), str(t.vo2_ratio_s2), str(t.vo2_max),
                            str(t.vo2_kg_max), str(t.ve_max), str(t.speed_s1), str(t.speed_s2), str(t.speed_max),
                            str(t.hr_max), str(t.lactate_max)] for t in chosen_tests[::-1]]

            # Set the x-axis labels
            x_label = 'Vitesse (km/h)'

            # Set the tabular label
            levels_labels = ["V S1", "V S2", "Vmax"]
            levels_units = "km/h"

        else :

            # Get the lactate, VO2, power associated threshold and tabular data
            lactate_curves = {t.date : [pd.read_json(StringIO(t.source_excel))['puissance'], pd.read_json(StringIO(t.source_excel))['lactate']] for t in chosen_tests}
            vo2_curves = {t.date : [pd.read_json(StringIO(t.source_excel))['puissance'], pd.read_json(StringIO(t.source_excel))['vo2']] for t in chosen_tests}
            power_values = pd.DataFrame([[t.date, t.watt_kg_s1, t.watt_kg_s2, t.watt_kg_max] for t in chosen_tests], columns = ["date", "s1", "s2", "max"])
            tabular_values = [[t.date.strftime("%d-%m-%Y"), str(t.weight), str(t.vo2_ratio_s1), str(t.vo2_ratio_s2), str(t.vo2_max),
                            str(t.vo2_kg_max), str(t.ve_max), str(t.watt_kg_s1), str(t.watt_kg_s2), str(t.watt_kg_max),
                            str(t.hr_max), str(t.lactate_max)] for t in chosen_tests[::-1]]

            # Set the x-axis labels
            x_label = 'Puissance (Watt)'

            # Set the tabular label
            levels_labels = ["P S1", "P S2", "Pmax"]
            levels_units = "Watt/kg"

        # Create the plot for the lactate & VO2 follow-up
        fig_values = make_subplots(
            rows=1, cols=2,
            horizontal_spacing=0.1,
            subplot_titles=(
                'Evolution des lactates',
                'Evolution de la VO2'
            )
        )

        # Create the colormaps
        cmap_lactate = plt.get_cmap('Reds')
        cmap_vo2 = plt.get_cmap('Blues')

        # Plot lactate and VO2
        dates = lactate_curves.keys()
        i = 0
        for d in dates :
            date_string = d.strftime("%d/%m/%Y")

            # Handling missing data
            no_data_lactate = 'N/A' if pd.isnull(sum(lactate_curves[d][1])) else ''
            no_data_vo2 = 'N/A' if pd.isnull(sum(vo2_curves[d][1])) else ''

            # Add VO2 max scatter plot trace
            fig_values.add_trace(
                go.Scatter(
                    x=lactate_curves[d][0], 
                    y=lactate_curves[d][1],
                    mode="lines+markers",
                    name = f"{date_string} {no_data_lactate}", 
                    line = dict(width=2),
                    marker = dict(color=f"rgba{cmap_lactate(0.2+0.8*(i/len(lactate_curves)))}", size=6, symbol='0'),
                    legendgroup=1,
                    legendgrouptitle_text="Lactate"
                ),
                row=1, col=1
            )

            # Add Lactate max scatter plot trace
            fig_values.add_trace(
                go.Scatter(
                    x=vo2_curves[d][0], 
                    y=vo2_curves[d][1],
                    mode="lines+markers",
                    name = f"{date_string} {no_data_vo2}", 
                    line = dict(width=2),
                    marker = dict(color=f"rgba{cmap_vo2(0.2+0.8*(i/len(lactate_curves)))}", size=6, symbol='0'),
                    legendgroup=2,
                    legendgrouptitle_text="VO2"
                ),
                row=1, col=2
            )

            # Increment the color index
            i+=1

        # Update global layout
        for i in range(1, 3):
            title = 'Lactate (mmol/L)' if i == 1 else 'VO2 (mL/min/kg)'
            fig_values.update_xaxes(
                title_text=x_label, 
                title_font=dict(size=12),
                row=1, col=i,
                tickfont=dict(size=10),
                showgrid=True
            )
            fig_values.update_yaxes(
                title_text=title,
                title_font=dict(size=12),
                row=1, col=i,
                tickfont=dict(size=10),
                showgrid=True,
            )
        fig_values.update_layout(
            legend=dict(
                font=dict(size=12),
                xanchor="left",
                yanchor="middle",
                x=1.05,
                y=0.5,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="LightGray",
                borderwidth=1,
            ),
            margin=dict(l=30, r=60, t=30, b=50),
            template="plotly_white",
            height=400
        )

        # Create the plot for the threshold data
        fig_thresholds = go.Figure()

        # Plot the evolution of the threshold values
        fig_thresholds.add_trace(
            go.Scatter(
                x=list(power_values["date"]),
                y=list(power_values["s1"]),
                mode="lines+markers",
                name="Seuil 1",
                line=dict(color="#58D68D"),
                marker=dict(symbol='0', size=6),
            )
        )
        fig_thresholds.add_trace(
            go.Scatter(
                x=list(power_values["date"]),
                y=list(power_values["s2"]),
                mode="lines+markers",
                name="Seuil 2",
                line=dict(color="#F8C471"),
                marker=dict(symbol='0', size=6),
            )
        )
        fig_thresholds.add_trace(
            go.Scatter(
                x=list(power_values["date"]),
                y=list(power_values["max"]),
                mode="lines+markers",
                name="Max",
                line=dict(color="#E59866"),
                marker=dict(symbol='0', size=6),
            )
        )

        # Update layout
        print([d.strftime("%d/%m/%Y") for d in power_values["date"]])
        fig_thresholds.update_layout(
            title=dict(
                text='Evolution des seuils',
                font=dict(size=15),
                x=0.5,
                y=0.95
            ),
            xaxis=dict(
                title="Date",
                titlefont=dict(size=12),
                tickfont=dict(size=10),
                tickvals=list(power_values["date"]),
                ticktext=[d.strftime("%d/%m/%Y") for d in power_values["date"]],
                showgrid=True,
            ),
            yaxis=dict(
                title=x_label,
                titlefont=dict(size=12),
                tickfont=dict(size=10),
                tickvals=np.round(np.linspace(power_values["s1"].min(), power_values["max"].max(), 12), 2),
                ticktext=np.round(np.linspace(power_values["s1"].min(), power_values["max"].max(), 12), 2),
                showgrid=True,
            ),
            legend=dict(
                font=dict(size=12),
                xanchor="left",
                yanchor="middle",
                x=1.05,
                y=0.5,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="LightGray",
                borderwidth=1,
            ),
            template="plotly_white",
            margin=dict(l=65, r=15, t=50, b=40),
            height=400
        )

        # Color the last test values to mark the evolution if there are multiple tests
        if len(tabular_values) > 1 :
            for i in range(1, len(tabular_values[0])) :
                if tabular_values[0][i] < tabular_values[1][i] :
                    tabular_values[0][i] = r"\textcolor{redmax}{"+tabular_values[0][i]+"}"
                elif tabular_values[0][i] > tabular_values[1][i] :
                    tabular_values[0][i] = r"\textcolor{greens1}{"+tabular_values[0][i]+"}"

        # Replace None values in the tabular
        tabular_values = [['N/A' if value == 'None' else value for value in subt] for subt in tabular_values]

        # Create the latex tabular
        latex_tabular_values = [r" & ".join(t) for t in tabular_values]
        latex_tabular_values = r" \\ \hline ".join(latex_tabular_values) + r" \\ \hline "

        return fig_values, fig_thresholds, latex_tabular_values, levels_labels, levels_units
    
    except Exception as e:
        logging.error(f"Error in create_followup_report: {e}")
        raise
