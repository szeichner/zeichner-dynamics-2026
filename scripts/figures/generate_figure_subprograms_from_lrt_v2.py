#!/usr/bin/env python3
"""
v2 — Narrative-driven subcategory assignments for k=3 programs.

Categories are organized around the biological story of each program
rather than generic cellular compartments:
  P1: Rapid bacterial growth response to free monomers
  P2: Bacterial stress/decline → fungal succession → necromass recycling
  P3: Constitutive community metabolism displaced by monomers

Reads data/derived/kmeans_program_assignments.tsv, classifies each KO,
and exports the annotated table for Figure 3 bubble heatmap.
"""

import os, re, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

warnings.filterwarnings('ignore')

BASE = '/Users/sarahzeichner/Documents/Berkeley/Soil OM complexity /claude/'
DERIVED_DIR = os.path.join(BASE, 'data', 'derived')
FIGURES_DIR = os.path.join(BASE, 'figures', 'output')
TABLES_DIR = os.path.join(BASE, 'tables', 'output')
FF = 'Arial'
MM = 1 / 25.4

PROG_COLORS = {
    'P_EARLY_MONO': '#6B7C2E',
    'P_LATE_MONO':  '#A2C060',
    'P_HEMI':       '#7B3F6E',
}

PROG_NAMES = {
    'P_EARLY_MONO': 'P1: Monomer growth\nresponse',
    'P_LATE_MONO':  'P2: Late succession &\nnecromass recycling',
    'P_HEMI':       'P3: Hemicellulose-\nassociated community',
}

PROG_ORDER = ['P_EARLY_MONO', 'P_LATE_MONO', 'P_HEMI']

# ============================================================================
# NARRATIVE-DRIVEN SUBCATEGORY DEFINITIONS (k=3)
# ============================================================================

P1_CATS = [
    # Core P1 narrative: rapid bacterial growth — proteostasis, redox, replication
    ('Proteostasis &\nstress response', [
        'chaperone', 'chaperonin', 'dnaj', 'dnak', 'grpe', 'groel', 'groes',
        'clp.*protease', 'clpb', 'clpc', 'clpe', 'clpa',
        'lon protease', 'ftsh', 'htpx', 'hslv', 'hslu',
        'heat.*shock', 'heat.*inducible', 'hsp20', 'ibpa',
        'cold.*shock', 'phage.*shock', 'stress',
        'serine protease do',
    ]),
    ('Thioredoxin &\nredox maintenance', [
        'thioredoxin', 'glutaredoxin', 'peroxiredoxin', 'mycoredoxin',
        'bacilliredoxin', 'methionine.*sulfoxide.*reductase',
        'n-ethylmaleimide reductase',
    ]),
    ('Fe-S cluster\nbiogenesis', [
        'fe.*s.*cluster', 'fe.*s.*biogenesis', 'sufb', 'sufc', 'sufd', 'sufe',
        'nifu', 'isca', 'cysteine desulfur',
    ]),
    ('TCA cycle &\ncentral carbon', [
        'citrate synthase', 'aconitase', 'aconitate',
        'isocitrate dehydrogenase', 'oxoglutarate.*dehydrogenase',
        '2-oxoglutarate', 'succinate dehydrogenase', 'succinyl-coa',
        'fumarase', 'fumarate hydratase', 'malate dehydrogenase',
        'pyruvate kinase', 'pyruvate dehydrogenase', 'dihydrolipoyl',
        'phosphoglycerate', 'enolase', 'glyceraldehyde-3-phosphate dehydrogenase',
        'fructose-bisphosphate aldolase', 'phosphofructokinase',
        'glucose-6-phosphate isomerase', 'transketolase', 'transaldolase',
        'pentose phosphate', 'glucose-6-phosphate dehydrogenase',
        '6-phosphogluconate dehydrogenase',
    ]),
    ('DNA replication\n& cell division', [
        'dna.*gyrase', 'dna.*replication', 'dna.*polymerase', 'dna.*helicase',
        'dna.*ligase', 'topoisomerase', 'excinuclease', 'recombinase',
        'cell division', 'ftsz', 'chromosome.*partition',
        'dna.*repair',
    ]),
    ('Sigma factors &\nregulators', [
        'sigma.*factor', 'sigma.*32', 'sigma.*70', 'anti-sigma',
        'rna.*polymerase', 'transcription.*regulator', 'transcriptional.*regulator',
        'two-component', 'response regulator', 'sensor.*kinase',
        'adenylate cyclase', 'diguanylate cyclase',
    ]),
    ('Aerobic respiration\n& ETC', [
        'nadh.*quinone', 'nadh.*ubiquinone', 'nadh:quinone',
        'quinol.*cytochrome', 'cytochrome.*oxidase',
        'atp synthase', 'electron.*transfer', 'ferredoxin',
        'rnf complex', 'energy.*coupling',
    ]),
    ('Fermentation &\noverflow metabolism', [
        'lactate dehydrogenase', 'alcohol dehydrogenase',
        'acetaldehyde dehydrogenase', 'acetate kinase', 'phosphotransacetylase',
        'pyruvate formate-lyase', 'glycyl radical',
        'citrate lyase', 'glyoxylate', 'malate synthase', 'isocitrate.*lyase',
        'methylcitrate', 'methylglyoxal', 'acetoin',
        'hydroxybutyrate', 'ethanolamine', 'propanediol',
        'microcompartment',
    ]),
    ('Cofactor\nbiosynthesis', [
        'nad biosynthesis', 'quinolinate', 'riboflavin', 'thiamine',
        'cobalamin', 'heme', 'siroheme', 'folate', 'biotin',
        'pyridoxal', 'pantothenate', 'coenzyme', 'menaquinone',
        'porphyrin', 'precorrin', 'lipoyl synthase',
    ]),
    ('ABC/PTS\ntransport', [
        'sugar transport', 'pts.*system', 'phosphotransferase system',
        'abc transport', 'mfs transporter', 'permease', 'solute carrier',
        'porin', 'outer membrane protein', 'tonb',
        'transport.*system', 'exporter', 'importer',
        'sulfonate', 'nitrate.*transport',
    ]),
    ('Lipid & cell\nenvelope', [
        'fatty acid', 'phospholipid', 'lipid', 'acyl.*carrier',
        'acyl-coa', 'enoyl-coa',
        'lps', 'lipopolysaccharide', 'cell wall', 'peptidoglycan', 'murein',
        'flagell', 'chemotaxis', 'pilus', 'adhesin',
        'multidrug', 'efflux', 'secretion',
    ]),
    ('Amino acid\nmetabolism', [
        'aminotransferase', 'amino.*acid', 'deaminase', 'racemase',
        'glutamate', 'aspartate', 'arginine', 'proline', 'lysine',
        'histidin', 'tryptophan', 'branched-chain',
        'nitrogen', 'ammonium', 'urea',
        'peptidase', 'protease',
    ]),
    ('RNA, ribosome &\ntranslation', [
        'ribosom', 'rrna', 'trna', 'translation', 'elongation factor',
        'rnase', 'ribonuclease', 'rna.*helicase',
    ]),
    ('Iron & metal\nhomeostasis', [
        'iron', 'siderophore', 'ferrous', 'ferric', 'zinc', 'copper',
        'cobalt', 'manganese', 'nickel', 'catalase', 'superoxide',
    ]),
]

P2_CATS = [
    # Core P2 narrative: bacterial decline → fungal succession → necromass recycling
    ('Bacterial stress &\nstationary phase', [
        'hsp20', 'ibpa', 'ibpb', 'molecular chaperone',
        'glutamate decarboxylase', 'superoxide reductase',
        'superoxide dismutase', 'cold.*shock', 'heat.*shock',
        'stress.*resistance', 'stationary.*phase',
        'mechanosensitive', 'acid.*resistance',
        'sporulation', 'spore.*coat', 'germination',
        'peptidoglycan.*hydrolase', 'n-acetylmuramoyl',
        'penicillin-binding',
    ]),
    ('Protein recycling\n& autophagy', [
        'ubiquitin', 'sumo', 'proteasome', 'protease', 'peptidase',
        'f-box', 'cullin', 'autophagy', 'vacuol',
        'lysosome', 'peroxisome', 'peroxin',
        'vesicle', 'golgi', 'endoplasmic reticulum', 'er membrane',
        'gpi.*anchor', 'exocyst', 'trafficking', 'sorting',
        'clathrin', 'dynamin', 'cargo.*transport',
    ]),
    ('Fungal cell wall\n& growth', [
        'glucan synthase', 'glucanosyl', 'chitin', 'cell wall.*integrity',
        'sterol', 'ergosterol', 'ergosteryl', 'mevalonate',
        'tubulin', 'actin', 'cytoskeleton', 'kinesin', 'dynein', 'myosin',
        'cell cycle', 'cyclin', 'cdc', 'separase', 'anaphase',
        'kinetochore', 'cohesin', 'mitosis', 'meiosis', 'spindle',
    ]),
    ('Fungal energy &\niron storage', [
        'mitochondri', 'f-type.*atp', 'cytochrome.*oxidase.*subunit',
        'complex.*i.*subunit', 'coenzyme.*q', 'nadh.*dehydrogenase.*ubiquinone',
        'ferricrocin', 'siderophore.*synthase',
        'copper chaperone', 'iron.*storage',
    ]),
    ('Chromatin\nremodeling', [
        'chromatin', 'histone', 'nucleosome', 'ino80', 'swi.*snf', 'rsc',
        'mediator', 'spt20', 'nuclear.*pore', 'nucleoporin',
        'spliceosome', 'rna.*splicing', 'rna.*processing',
        'transcription.*initiation', 'transcription.*elongation',
        'polyadenylation', 'mrna.*decapping',
    ]),
    ('Necromass N\nscavenging', [
        'dal5', 'allantoin', 'allantoate', 'allantoicase',
        'uricase', 'xanthine', 'purine',
        'amino acid transporter.*yeast', 'yeast amino acid',
        'glutamate synthase', 'glutamine synthetase',
        'nitrogen', 'ammonium', 'urea',
        'nucleoside', 'nucleotide',
        'choline', 'betaine', 'carnitine',
    ]),
    ('Two-component &\ntranscriptional regulation', [
        'transcription.*regulator', 'transcriptional.*regulator',
        'two-component', 'response regulator', 'sensor.*kinase',
        'sigma.*factor', 'dna-binding',
        'dna.*replication', 'dna.*repair',
    ]),
    ('Central carbon\n& cofactors', [
        'dehydrogenase', 'reductase', 'synthase', 'transferase',
        'oxidase', 'kinase', 'ligase', 'decarboxylase', 'lyase',
        'isomerase', 'mutase', 'epimerase',
        'amino.*acid', 'aminotransferase',
        'sugar', 'glycerol', 'propionate', 'aromatic',
        'fatty acid', 'phospholipid', 'acyl-coa',
        'folate', 'cobalamin', 'vitamin',
    ]),
    ('Membrane transport\n& efflux', [
        'transport', 'permease', 'channel', 'porin', 'exporter',
        'importer', 'membrane', 'abc', 'mfs',
        'pts.*system', 'phosphotransferase',
        'efflux', 'multidrug',
    ]),
]

P3_CATS = [
    # Core P3 narrative: constitutive community — surface attachment, polymer degradation, baseline metabolism
    ('Surface attachment\n& secretion', [
        'pilus', 'type.*iv.*pilus', 'cpac', 'cpad', 'cpaf', 'tad.*pilus',
        'type.*ii.*secretion', 'type.*vi.*secretion',
        'general secretion pathway', 't2ss', 't6ss',
        'adhesin', 'fibronectin', 'gliding.*motility', 'twitching',
    ]),
    ('TonB-dependent\ntransport', [
        'tonb', 'tolr', 'tolb', 'exbb', 'exbd',
        'outer membrane.*protein.*insertion', 'bamb', 'bama',
        'outer membrane.*assembly',
    ]),
    ('CAZymes &\npolymer hydrolysis', [
        'cellulase', 'cellobiosidase', 'endoglucanase', 'xylanase',
        'xylosidase', 'glucosidase', 'galactosidase', 'mannosidase',
        'mannanase', 'fucosidase', 'arabinofuranohydrolase', 'pectate',
        'polygalacturonase', 'rhamnogalacturonan', 'glucuronoarabinoxylan',
        'xyloglucan', 'starch.*binding', 'susc', 'susd', 'suse',
        'dextranase', 'amylase', 'pullulanase', 'laminarinase',
        'chitinase', 'chitin.*deacetylase',
        'glycoside hydrolase', 'polysaccharide lyase',
        'glucuronoyl esterase', 'glucuronate.*lignin',
    ]),
    ('Aromatic & lignin\ncatabolism', [
        'catechol', 'protocatechuate', 'lignin', 'phenol.*monooxygenase',
        'toluene', 'benzoate', 'vanillate', 'aromatic', 'dioxygenase',
        'gentisate', 'homogentisate', 'hydroxyphenylpyruvate',
        'dihydroxyphthalate',
    ]),
    ('Amino acid catabolism\n& proteolysis', [
        'peptidase', 'protease', 'proteinase', 'aminopeptidase',
        'carboxypeptidase', 'dipeptidyl', 'prolyl oligopeptidase',
        'tricorn protease', 'bleomycin hydrolase',
        'deaminase', 'racemase',
        'amino.*acid.*transport', 'branched-chain',
    ]),
    ('Carbohydrate transport\n& phosphorylation', [
        'glucose.*dehydrogenase', 'aldose.*dehydrogenase',
        'sugar transport', 'sugar.*permease',
        'glucokinase', 'ribulokinase',
        'pts.*system', 'phosphotransferase.*system',
        'myo-inositol', 'inositol',
    ]),
    ('Iron & metal\nacquisition', [
        'iron', 'siderophore', 'ferri', 'ferrous',
        'cobalt', 'manganese', 'zinc', 'copper', 'nickel',
        'arylsulfatase',
    ]),
    ('Cell envelope,\nLPS & hopanoids', [
        'hopanoid', 'menaquinone', 'lipopolysaccharide', 'lps',
        'cell wall', 'peptidoglycan', 'murein', 'fatty acid',
        'phospholipid', 'multidrug', 'efflux',
        'antibiotic.*resistance', 'beta-lactamase',
    ]),
    ('Two-component\nsignaling', [
        'transcription.*regulator', 'transcriptional.*regulator',
        'two-component', 'response regulator', 'sensor.*kinase',
        'sigma.*factor', 'circadian',
    ]),
    ('Electron transport &\nalternative respiration', [
        'cytochrome', 'quinol', 'nadh', 'ferredoxin', 'electron.*transfer',
        'atp synthase', 'formate dehydrogenase',
    ]),
    ('N assimilation &\nbiosynthesis', [
        'glutamine synthetase', 'glutamate synthase', 'asparagine synthase',
        'chorismate', 'prephenate', 'diaminopimelate', 'anthranilate',
        'histidinol', 'nitrogen fix', 'nitrogenase', 'nif',
        'nitrate', 'nitrite', 'ammonium',
        'arginine decarboxylase', 'polyamine',
    ]),
    ('DNA, RNA &\ntranslation', [
        'ribosom', 'trna', 'translation', 'dna.*repair', 'dna.*replication',
        'dna.*polymerase', 'rna.*helicase', 'rnase',
        'restriction', 'methyltransferase',
    ]),
]

SUBCAT_DEFS = {
    'P_EARLY_MONO': P1_CATS,
    'P_LATE_MONO': P2_CATS,
    'P_HEMI': P3_CATS,
}

# ============================================================================
# Load and classify
# ============================================================================
print("Loading k-means program assignments...")
df = pd.read_csv(os.path.join(DERIVED_DIR, 'kmeans_program_assignments.tsv'), sep='\t')

mono_cols = [c for c in df.columns if c.startswith('MONOMER_')]
hemi_cols = [c for c in df.columns if c.startswith('HEMICELLULOSE_')]
all_data_cols = mono_cols + hemi_cols

print(f"  {len(df)} KOs across {df.mnemonic.nunique()} programs")

def classify_ko(desc, prog):
    if prog not in SUBCAT_DEFS:
        return 'Other/diverse'
    desc_lower = str(desc).lower()
    for cat_label, patterns in SUBCAT_DEFS[prog]:
        for pat in patterns:
            if re.search(pat, desc_lower):
                return cat_label
    return 'Other/diverse'

df['subcategory'] = df.apply(lambda r: classify_ko(r.ko_description, r.mnemonic), axis=1)

print("\nSubcategory classification (v2, narrative-driven):")
for prog in PROG_ORDER:
    sub = df[df.mnemonic == prog]
    print(f"\n  {PROG_NAMES[prog].replace(chr(10), ' ')} (n={len(sub)}):")
    cats = SUBCAT_DEFS[prog]
    cat_order = [c[0] for c in cats] + ['Other/diverse']
    for cat in cat_order:
        n = (sub.subcategory == cat).sum()
        if n > 0:
            print(f"    {n:4d}  {cat.replace(chr(10), ' ')}")

# Save full table
out_tsv = os.path.join(TABLES_DIR, 'TableS9_program_subcategory_assignments_v2.tsv')
df_out = df.copy()
df_out['subcategory_flat'] = df_out.subcategory.str.replace('\n', ' ')
df_out.to_csv(out_tsv, sep='\t', index=False)
print(f"\nSaved: {out_tsv}")
