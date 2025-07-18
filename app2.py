import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="🎮 StatPixel - Tests Statistiques",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS pour un style moderne et lisible
st.markdown("""
<style>
    .main-header {
        font-size: 28px;
        color: #2c3e50;
        text-align: center;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 25px;
        border-radius: 15px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        font-weight: bold;
    }
    
    .section-container {
        background: white;
        border-radius: 12px;
        padding: 20px;
        margin: 20px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #3498db;
    }
    
    .result-box {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        border: 1px solid #dee2e6;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .result-box h3 {
        color: #2c3e50;
        margin-bottom: 15px;
        font-size: 18px;
        border-bottom: 2px solid #3498db;
        padding-bottom: 5px;
    }
    
    .stats-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin: 20px 0;
    }
    
    .stat-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e9ecef;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        transition: transform 0.2s;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .stat-card h4 {
        color: #6c757d;
        font-size: 12px;
        margin-bottom: 8px;
        text-transform: uppercase;
        font-weight: 600;
    }
    
    .stat-card .value {
        color: #2c3e50;
        font-size: 16px;
        font-weight: bold;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3498db 0%, #2980b9 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(52, 152, 219, 0.3);
    }
    
    .success-indicator {
        color: #27ae60;
        font-weight: bold;
    }
    
    .error-indicator {
        color: #e74c3c;
        font-weight: bold;
    }
    
    .section-title {
        color: #2c3e50;
        font-size: 20px;
        font-weight: bold;
        margin-bottom: 15px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Classes pour les tests statistiques
class StatisticalTests:
    def __init__(self):
        self.results = {}
    
    def descriptive_stats(self, data):
        """Calcul des statistiques descriptives"""
        stats_dict = {}
        for col in data.select_dtypes(include=[np.number]).columns:
            values = data[col].dropna()
            stats_dict[col] = {
                'count': len(values),
                'mean': np.mean(values),
                'median': np.median(values),
                'mode': stats.mode(values, keepdims=True)[0][0] if len(values) > 0 else np.nan,
                'std': np.std(values, ddof=1),
                'var': np.var(values, ddof=1),
                'q1': np.percentile(values, 25),
                'q3': np.percentile(values, 75),
                'min': np.min(values),
                'max': np.max(values)
            }
        return stats_dict
    
    def mann_whitney_test(self, group1, group2):
        """Test de Mann-Whitney U"""
        statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
        return {
            'test': 'Mann-Whitney U',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': 'Différence significative' if p_value < 0.05 else 'Pas de différence significative'
        }
    
    def wilcoxon_test(self, group1, group2):
        """Test de Wilcoxon (échantillons appariés)"""
        statistic, p_value = stats.wilcoxon(group1, group2)
        return {
            'test': 'Wilcoxon',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': 'Différence significative' if p_value < 0.05 else 'Pas de différence significative'
        }
    
    def student_test(self, group1, group2, paired=False):
        """Test de Student"""
        if paired:
            statistic, p_value = stats.ttest_rel(group1, group2)
            test_name = 'Student (apparié)'
        else:
            statistic, p_value = stats.ttest_ind(group1, group2)
            test_name = 'Student (indépendant)'
        
        return {
            'test': test_name,
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': 'Différence significative' if p_value < 0.05 else 'Pas de différence significative'
        }
    
    def welch_test(self, group1, group2):
        """Test de Welch (Student avec variances inégales)"""
        statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        return {
            'test': 'Welch',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': 'Différence significative' if p_value < 0.05 else 'Pas de différence significative'
        }
    
    def anova_test(self, *groups):
        """Test ANOVA à un facteur"""
        statistic, p_value = stats.f_oneway(*groups)
        return {
            'test': 'ANOVA',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': 'Différence significative entre les groupes' if p_value < 0.05 else 'Pas de différence significative'
        }
    
    def chi2_test(self, contingency_table):
        """Test du Chi²"""
        statistic, p_value, dof, expected = stats.chi2_contingency(contingency_table)
        return {
            'test': 'Chi²',
            'statistic': statistic,
            'p_value': p_value,
            'dof': dof,
            'significant': p_value < 0.05,
            'interpretation': 'Association significative' if p_value < 0.05 else 'Pas d\'association significative'
        }
    
    def mcnemar_test(self, table):
        """Test de McNemar"""
        # Table 2x2 pour échantillons appariés
        statistic, p_value = stats.mcnemar(table, exact=False)
        return {
            'test': 'McNemar',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'interpretation': 'Changement significatif' if p_value < 0.05 else 'Pas de changement significatif'
        }
    
    def suggest_test(self, data, data_type, paired=False, num_groups=2):
        """Suggère le test le plus approprié"""
        if data_type == 'contingency':
            if data.shape == (2, 2) and paired:
                return 'McNemar'
            else:
                return 'Chi²'
        
        elif data_type == 'continuous':
            # Test de normalité
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) >= 2:
                col1, col2 = numeric_cols[0], numeric_cols[1]
                group1 = data[col1].dropna()
                group2 = data[col2].dropna()
                
                # Test de normalité de Shapiro-Wilk
                _, p1 = stats.shapiro(group1) if len(group1) <= 5000 else (0, 0.001)
                _, p2 = stats.shapiro(group2) if len(group2) <= 5000 else (0, 0.001)
                
                normal_dist = p1 > 0.05 and p2 > 0.05
                
                if num_groups == 2:
                    if normal_dist:
                        if paired:
                            return 'Student (apparié)'
                        else:
                            # Test d'égalité des variances
                            _, p_var = stats.levene(group1, group2)
                            if p_var > 0.05:
                                return 'Student (indépendant)'
                            else:
                                return 'Welch'
                    else:
                        if paired:
                            return 'Wilcoxon'
                        else:
                            return 'Mann-Whitney'
                else:
                    if normal_dist:
                        return 'ANOVA'
                    else:
                        return 'Kruskal-Wallis'
        
        return 'Test non déterminé'

# Fonction pour générer le PDF
def generate_pdf(results, descriptive_stats, test_results):
    """Génère un rapport PDF des résultats"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    
    # Style personnalisé
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=30,
        textColor=colors.darkblue
    )
    
    story = []
    
    # Titre
    story.append(Paragraph("🎮 StatPixel - Rapport d'Analyse Statistique", title_style))
    story.append(Spacer(1, 20))
    
    # Date
    story.append(Paragraph(f"Date d'analyse: {datetime.now().strftime('%d/%m/%Y %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Statistiques descriptives
    if descriptive_stats:
        story.append(Paragraph("📊 Statistiques Descriptives", styles['Heading2']))
        for col, stats_dict in descriptive_stats.items():
            story.append(Paragraph(f"Variable: {col}", styles['Heading3']))
            data = [
                ['Statistique', 'Valeur'],
                ['Nombre d\'observations', f"{stats_dict['count']}"],
                ['Moyenne', f"{stats_dict['mean']:.4f}"],
                ['Médiane', f"{stats_dict['median']:.4f}"],
                ['Mode', f"{stats_dict['mode']:.4f}"],
                ['Écart-type', f"{stats_dict['std']:.4f}"],
                ['Variance', f"{stats_dict['var']:.4f}"],
                ['Q1', f"{stats_dict['q1']:.4f}"],
                ['Q3', f"{stats_dict['q3']:.4f}"],
                ['Minimum', f"{stats_dict['min']:.4f}"],
                ['Maximum', f"{stats_dict['max']:.4f}"]
            ]
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 20))
    
    # Résultats des tests
    if test_results:
        story.append(Paragraph("🧪 Résultats des Tests Statistiques", styles['Heading2']))
        for result in test_results:
            story.append(Paragraph(f"Test: {result['test']}", styles['Heading3']))
            story.append(Paragraph(f"Statistique: {result['statistic']:.4f}", styles['Normal']))
            story.append(Paragraph(f"P-value: {result['p_value']:.4f}", styles['Normal']))
            story.append(Paragraph(f"Significatif (α=0.05): {'Oui' if result['significant'] else 'Non'}", styles['Normal']))
            story.append(Paragraph(f"Interprétation: {result['interpretation']}", styles['Normal']))
            story.append(Spacer(1, 15))
    
    doc.build(story)
    buffer.seek(0)
    return buffer

# Interface principale
def main():
    # Titre principal
    st.markdown('<div class="main-header">🎮 StatPixel - Analyseur Statistique</div>', unsafe_allow_html=True)
    
    # Initialisation de la classe de tests
    stat_tests = StatisticalTests()
    
    # Sidebar pour les paramètres
    st.sidebar.markdown('<div class="section-title">⚙️ Paramètres</div>', unsafe_allow_html=True)
    
    # Type de données
    data_type = st.sidebar.selectbox(
        "Type de données",
        ["Données continues", "Tableau de contingence"],
        key="data_type"
    )
    
    # Données appariées
    paired = st.sidebar.checkbox("Données appariées", value=False)
    
    # Upload de fichier
    st.markdown('<div class="section-container"><div class="section-title">📁 Chargement des données</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choisissez un fichier",
        type=['csv', 'xlsx', 'xls'],
        help="Formats supportés: CSV, Excel"
    )
    
    if uploaded_file is not None:
        try:
            # Lecture du fichier
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Fichier chargé: {uploaded_file.name}")
            
            # Affichage des données
            st.markdown('<div class="section-title">🔍 Aperçu des données</div>', unsafe_allow_html=True)
            st.dataframe(df.head(), use_container_width=True)
            
            # Informations sur les données
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f'<div class="stat-card"><h4>Lignes</h4><div class="value">{len(df)}</div></div>', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<div class="stat-card"><h4>Colonnes</h4><div class="value">{len(df.columns)}</div></div>', unsafe_allow_html=True)
            with col3:
                st.markdown(f'<div class="stat-card"><h4>Valeurs manquantes</h4><div class="value">{df.isnull().sum().sum()}</div></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Statistiques descriptives
            if data_type == "Données continues":
                st.markdown('<div class="section-container"><div class="section-title">📊 Statistiques descriptives</div>', unsafe_allow_html=True)
                
                descriptive_stats = stat_tests.descriptive_stats(df)
                
                for col, stats_dict in descriptive_stats.items():
                    st.markdown(f"**📈 Variable: {col}**")
                    
                    # Grille de statistiques
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.markdown(f'<div class="stat-card"><h4>Moyenne</h4><div class="value">{stats_dict["mean"]:.4f}</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="stat-card"><h4>Médiane</h4><div class="value">{stats_dict["median"]:.4f}</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="stat-card"><h4>Mode</h4><div class="value">{stats_dict["mode"]:.4f}</div></div>', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'<div class="stat-card"><h4>Écart-type</h4><div class="value">{stats_dict["std"]:.4f}</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="stat-card"><h4>Variance</h4><div class="value">{stats_dict["var"]:.4f}</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="stat-card"><h4>Observations</h4><div class="value">{stats_dict["count"]}</div></div>', unsafe_allow_html=True)
                    with col3:
                        st.markdown(f'<div class="stat-card"><h4>Q1</h4><div class="value">{stats_dict["q1"]:.4f}</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="stat-card"><h4>Q3</h4><div class="value">{stats_dict["q3"]:.4f}</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="stat-card"><h4>Minimum</h4><div class="value">{stats_dict["min"]:.4f}</div></div>', unsafe_allow_html=True)
                    with col4:
                        st.markdown(f'<div class="stat-card"><h4>Maximum</h4><div class="value">{stats_dict["max"]:.4f}</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="stat-card"><h4>Étendue</h4><div class="value">{stats_dict["max"] - stats_dict["min"]:.4f}</div></div>', unsafe_allow_html=True)
                        st.markdown(f'<div class="stat-card"><h4>IQR</h4><div class="value">{stats_dict["q3"] - stats_dict["q1"]:.4f}</div></div>', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Tests statistiques
            st.markdown('<div class="section-container"><div class="section-title">🧪 Tests statistiques</div>', unsafe_allow_html=True)
            
            # Sélection des colonnes pour les tests
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if len(numeric_cols) >= 2:
                col1, col2 = st.columns(2)
                with col1:
                    group1_col = st.selectbox("Groupe 1", numeric_cols, key="group1")
                with col2:
                    group2_col = st.selectbox("Groupe 2", numeric_cols, key="group2", index=1 if len(numeric_cols) > 1 else 0)
                
                # Suggestion automatique de test
                suggested_test = stat_tests.suggest_test(df, 'continuous', paired, 2)
                st.info(f"🤖 **Test suggéré :** {suggested_test}")
                
                # Boutons pour les tests
                st.markdown("**Sélectionnez un test à effectuer :**")
                test_col1, test_col2, test_col3 = st.columns(3)
                
                test_results = []
                
                with test_col1:
                    if st.button("📊 Mann-Whitney"):
                        group1 = df[group1_col].dropna()
                        group2 = df[group2_col].dropna()
                        result = stat_tests.mann_whitney_test(group1, group2)
                        test_results.append(result)
                    
                    if st.button("📊 Student"):
                        group1 = df[group1_col].dropna()
                        group2 = df[group2_col].dropna()
                        result = stat_tests.student_test(group1, group2, paired)
                        test_results.append(result)
                    
                    if st.button("📊 Welch"):
                        group1 = df[group1_col].dropna()
                        group2 = df[group2_col].dropna()
                        result = stat_tests.welch_test(group1, group2)
                        test_results.append(result)
                
                with test_col2:
                    if st.button("📊 Wilcoxon") and paired:
                        group1 = df[group1_col].dropna()
                        group2 = df[group2_col].dropna()
                        if len(group1) == len(group2):
                            result = stat_tests.wilcoxon_test(group1, group2)
                            test_results.append(result)
                        else:
                            st.error("Les groupes doivent avoir la même taille pour le test de Wilcoxon")
                    
                    if st.button("📊 ANOVA") and len(numeric_cols) >= 3:
                        groups = [df[col].dropna() for col in numeric_cols[:3]]
                        result = stat_tests.anova_test(*groups)
                        test_results.append(result)
                
                with test_col3:
                    if st.button("📊 Chi²") and data_type == "Tableau de contingence":
                        if df.shape[0] >= 2 and df.shape[1] >= 2:
                            result = stat_tests.chi2_test(df.iloc[:, :2])
                            test_results.append(result)
                    
                    if st.button("📊 McNemar") and paired and df.shape == (2, 2):
                        result = stat_tests.mcnemar_test(df)
                        test_results.append(result)
                
                # Affichage des résultats
                if test_results:
                    st.markdown('<div class="section-title">📋 Résultats des tests</div>', unsafe_allow_html=True)
                    
                    for result in test_results:
                        status_class = "success-indicator" if result['significant'] else "error-indicator"
                        status_text = "✅ SIGNIFICATIF" if result['significant'] else "❌ NON SIGNIFICATIF"
                        
                        st.markdown(f'''
                        <div class="result-box">
                            <h3>📊 {result['test']}</h3>
                            <p><strong>Statistique :</strong> {result['statistic']:.4f}</p>
                            <p><strong>P-value :</strong> {result['p_value']:.4f}</p>
                            <p class="{status_class}">{status_text}</p>
                            <p><strong>Interprétation :</strong> {result['interpretation']}</p>
                        </div>
                        ''', unsafe_allow_html=True)
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Génération du PDF
                st.markdown('<div class="section-container"><div class="section-title">📄 Export PDF</div>', unsafe_allow_html=True)
                
                if st.button("📥 Générer le rapport PDF"):
                    with st.spinner("Génération du rapport..."):
                        pdf_buffer = generate_pdf(
                            {},
                            descriptive_stats if data_type == "Données continues" else {},
                            test_results
                        )
                        
                        st.download_button(
                            label="💾 Télécharger le rapport PDF",
                            data=pdf_buffer,
                            file_name=f"rapport_statistiques_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="application/pdf"
                        )
                
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.warning("⚠️ Il faut au moins 2 colonnes numériques pour effectuer des tests statistiques")
        
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du fichier: {str(e)}")
    
    else:
        st.info("👆 Veuillez charger un fichier pour commencer l'analyse")
        
        # Exemple de données
        st.markdown('<div class="section-container"><div class="section-title">💡 Exemple de données</div>', unsafe_allow_html=True)
        example_data = pd.DataFrame({
            'Groupe_A': np.random.normal(50, 10, 30),
            'Groupe_B': np.random.normal(55, 12, 30),
            'Groupe_C': np.random.normal(48, 9, 30)
        })
        st.dataframe(example_data.head(), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()