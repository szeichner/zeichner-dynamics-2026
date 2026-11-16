#!/usr/bin/env python3
"""
Generate subprogram heatmap from the spline-LRT × k-means analysis (v6).

Reads data/derived/kmeans_program_assignments.tsv (2,959 KOs from the
R DESeq2 VST + OLS F-test pipeline), classifies each KO into functional
subcategories using keyword matching on KO descriptions, computes mean
LFC per subcategory, and generates the page-sized heatmap figure.

Also exports the full annotated TSV for supplementary tables.
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
# Subcategory definitions: keyword patterns matched against ko_description
# ============================================================================

P1_CATS = [
    ('Central carbon &\noverflow metabolism', [
        'pyruvate kinase', 'pyruvate dehydrogenase', 'dihydrolipoyl',
        'phosphogluconate dehydratase', 'gluconokinase', 'phosphofructokinase',
        'glycogen', 'mannitol.*dehydrogenase', 'inositol', 'galactonate',
        'methylglyoxal', 'acetaldehyde dehydrogenase', 'oxalate',
        'acetoacetate decarboxylase', 'fructose', 'gluconate', 'amino sugar',
        'citrate lyase', 'atp citrate', 'malate dehydrogenase',
        'isocitrate dehydrogenase', 'succinate dehydrogenase',
        'fumarate', 'pentose phosphate', 'transketolase', 'transaldolase',
        'glucose-6-phosphate', 'glyceraldehyde-3-phosphate', 'phosphoglycerate',
        'glycerol dehydrogenase', 'glycerol.*kinase', 'glycerol uptake',
        'lactate dehydrogenase', 'malonate', 'oxoglutarate.*dehydrogenase',
        'oxoisovalerate.*dehydrogenase', 'deoxyribose.*aldolase',
        'levansucrase', 'glucosylglycerate', 'ectoine', 'acetyl-coa.*c-acetyltransferase',
    ]),
    ('Glyoxylate, C2/C3 &\nmicrocompartments', [
        'glyoxylate', 'malate synthase', 'isocitrate.*lyase', 'methylisocitrate',
        'methylcitrate', 'acetate.*coa.*transferase', 'acetoacetate.*coa',
        '3-oxoacid.*coa', 'ethylmalonyl', 'crotonyl-coa carboxylase',
        'microcompartment', 'propanediol dehydratase', 'crotonyl-coa reductase',
        'hydroxymethylglutaryl-coa', 'acetoin.*utilization',
        'succinyl-coa.*transferase',
    ]),
    ('Beta-oxidation, lipid\n& energy metabolism', [
        'acyl-coa dehydrogenase', 'acetyl-coa acyltransferase',
        'enoyl-coa', 'fatty acid synthase', 'phospholipid', 'sphingolipid',
        'sterol', 'glycerolipid', 'glycerol-3-phosphate dehydrogenase',
        'oxylipin', 'lipoxygenase', 'phospholipase', 'lysophospholipase',
        'oxidoreductase.*quinone', 'quinol.*cytochrome', 'cytochrome.*oxidase',
        'nadh.*quinone', 'nadh.*ubiquinone', 'electron.*transferring.*flavoprotein',
        'ferredoxin', 'rnf complex', 'energy.*coupling',
        'fatty acid.*coa.*ligase', 'fatty acid.*amide', 'acyl.*carrier.*protein',
        'oxoacyl.*reductase', 'malonyl.*transferase', 'cdp-diacylglycerol',
        'cdp-glycerol', 'glycerophospho', 'acyltransferase',
        'dihydroceramide', 'cholestenol',
    ]),
    ('Aromatic catabolism', [
        'catechol', 'phenol.*monooxygenase', 'toluene', 'benzoate',
        'phenylacet', 'aromatic', 'hydroxymuconate', 'p-cumate',
        'protocatechuate', 'gentisate', 'homogentisate',
        'hydroxyphenylpyruvate', 'indolepyruvate', 'indoleamine', 'quercetin',
        'oxalocrotonate', 'benzaldehyde',
    ]),
    ('Amino acid\nmetabolism', [
        'aminotransferase', 'amino.*acid.*racemase', 'deaminase',
        'glutamate.*dehydrogenase', 'glutamate.*synthase', 'glutamine.*synthetase',
        'glutamate.*5-kinase', 'glutamate-5-semialdehyde', 'proline.*dehydrogenase',
        '1-pyrroline-5-carboxylate', 'alanine.*dehydrogenase',
        'aspartate.*kinase', 'aspartate.*1-decarboxylase', 'asparagine',
        'homoserine', 'threonine', 'cystathionine',
        'glycine cleavage', 'glycine hydroxymethyl', 'serine hydroxymethyl',
        'diaminobutyrate', '2-aminoadipate', 'aminopeptidase',
        'isopropylmalate', 'branched-chain', 'oxoisovalerate',
        'd-serine.*dehydratase', 'amino-acid.*racemase',
    ]),
    ('N, S & cofactor\nbiosynthesis', [
        'sulfate adenylyltransferase', 'cysn', 'cysc', 'cysd',
        'aspartate oxidase', 'amine oxidase', 'succinylornithine',
        'arginine deiminase', 'ornithine', 'siderophore',
        'choline dehydrogenase', 'betaine.*aldehyde', 'methionine',
        'dimethylglycine', 'sarcosine', 'gaba', 'glutamate decarboxylase',
        'nad biosynthesis', 'quinolinate', 'riboflavin', 'thiamine',
        'cobalamin', 'heme', 'siroheme', 'mep.*isoprenoid', 'folate',
        'shikimate', 'chorismate', 'prephenate', 'anthranilate',
        'cysteine desulfur', 'sufb', 'sufc', 'sufd', 'sufe',
        'sulfur', 'cysteine', 'desulfurase',
        'pyridoxal', 'biotin.*synth', 'biotin.*transport', 'menaquinone',
        'pantothenate', 'coenzyme.*f420', 'nicotinate', 'nicotinamide',
        'porphyrin', 'precorrin', 'cobalt.*chelatase',
    ]),
    ('Iron, metal &\nredox homeostasis', [
        'iron.*transport', 'iron.*uptake', 'ferrous', 'ferric', 'iron.*complex',
        'manganese', 'zinc', 'copper', 'nickel', 'cobalt.*transport',
        'chromate', 'p-type.*cu', 'cation.*transport',
        'peroxiredoxin', 'glutaredoxin', 'glutathione', 'catalase',
        'superoxide.*dismutase', 'thioredoxin', 'mycoredoxin', 'bacilliredoxin',
        'fe.*s.*cluster', 'fe.*s.*biogenesis', 'nifu',
    ]),
    ('RNA, ribosome &\nnucleotide metabolism', [
        'rrna', 'ribosom', 'rna.*helicase', 'rnase', 'ribonuclease',
        'trna.*synthetase', 'aminoacyl.*trna', 'trna.*pseudouridine',
        'trna.*methyltransferase', 'translation.*factor', 'elongation factor',
        'imp.*dehydrogenase', 'gmp.*synthase', 'gmp.*reductase',
        'ctp.*synthase', 'nucleotid', 'purine', 'pyrimidine',
        'phosphoribosyl', 'amidophosphoribosyl',
    ]),
    ('Transport & PTS', [
        'sugar transport', 'pts.*ei', 'pts.*system', 'phosphotransferase system',
        'abc transport', 'mfs transporter', 'permease', 'solute carrier',
        'porin', 'outer membrane protein', 'tonb',
        'transport.*system.*atp', 'transport.*system.*substrate',
        'transport.*system.*permease', 'exporter', 'importer',
    ]),
    ('Motility, cell surface\n& efflux', [
        'flagell', 'chemotaxis', 'pilus', 'fimb', 'adhesin', 'adhesion',
        'lps', 'lipopolysaccharide', 'cell wall', 'peptidoglycan', 'murein',
        'cell envelope', 'multidrug', 'drug.*resistance', 'efflux',
        'secretion.*protein', 'preprotein translocase',
        'leader peptidase', 'lipoprotein-releasing',
    ]),
    ('Stress, regulation\n& protein QC', [
        'chaperone', 'dnak', 'dnaj', 'grpe', 'groel', 'groes', 'clp.*protease',
        'heat.*shock', 'heat.*inducible', 'htpx', 'stress', 'cold.*shock', 'osmo',
        'phage.*shock', 'antitoxin',
        'transcription.*regulator', 'transcriptional.*regulator',
        'two-component', 'response regulator', 'sensor.*kinase',
        'sigma.*factor', 'anti-sigma', 'adenylate cyclase',
        'diguanylate cyclase', 'c-di-', 'lon.*protease', 'hsl.*protease',
        'protease.*subunit',
    ]),
    ('Cell division &\nDNA maintenance', [
        'cell division', 'fts', 'chromosome.*partition', 'topoisomerase',
        'dna.*replication', 'dna.*repair', 'dna.*polymerase', 'dna.*helicase',
        'dna.*ligase', 'dna.*gyrase', 'excinuclease', 'recombinase',
        'nucleotide.*reductase', 'nucleoid',
    ]),
    ('Fungal/eukaryotic\nsignatures', [
        'fungi', 'fungal', 'eukaryot', 'kinetochore', 'histone',
        'chromatin', 'nuclear.*pore', 'nucleoporin', 'spliceosome',
        'polyketide', 'nonribosomal peptide', 'melanin', 'mycovirus',
        'partitivirus', 'apoptosis', 'caspase',
    ]),
]

P2_CATS = [
    ('PTS & sugar\nuptake/scavenging', [
        'pts.*ei', 'pts.*system', 'phosphotransferase system',
        'mannose.*pts', 'trehalose.*pts', 'sucrose.*pts', 'mannitol.*pts',
        'galactosamine.*pts', 'glucoside.*pts', 'fructose.*pts',
        'sugar transport', 'maltose', 'trehalose.*hydrolase',
        'lactaldehyde reductase', 'oligogalacturonide', 'pectin',
        'glucanotransferase', 'arabinooligosaccharide',
        'methyl-galactoside', 'glucosidase', 'phospho-beta-glucosidase',
        'galactose.*symporter', 'l-cystine.*transport',
    ]),
    ('Anaerobic fermentation\n& respiration', [
        'lactate dehydrogenase', 'butanol dehydrogenase', 'formate.*dehydrogenase',
        'formate.*hydrogenlyase', 'hydrogenase', 'dmso reductase',
        'nitrate reductase', 'nitrite reductase', 'selenate reductase',
        'urocanate reductase', 'proline reductase', 'stickland',
        'glycyl radical', 'ferredoxin.*nad.*oxidoreductase',
        'electron transfer', 'methylaspartate.*ammonia',
        'isopropanol dehydrogenase', 'acetoin', 'ethanolamine',
        'glycine.*reductase', 'betaine.*reductase',
        'nitric oxide reductase', 'succinyl-coa.*transferase',
        'citrate lyase', 'citrate.*transport', 'ketopentanoate.*thiolase',
    ]),
    ('Chemotaxis, competence\n& quorum sensing', [
        'methyl-accepting chemotaxis', 'chemotaxis', 'cheA', 'cheB', 'cheR',
        'competence.*com', 'quorum.*sensing', 'autoinducer', 'luxr',
    ]),
    ('Stress, sporulation\n& stationary phase', [
        'sporulation', 'spore.*coat', 'dipicolinate', 'germination',
        'acid-soluble.*spore', 'sasp',
        'stress.*resistance', 'stationary.*phase', 'ribosome.*inhibitor',
        'mechanosensitive', 'cold.*shock', 'heat.*shock',
        'outer membrane.*pore', 'penicillin-binding', 'multiple.*stress',
        'chaperone', 'flotillin', 'molecular.*chaperone',
    ]),
    ('Regulation &\nsignaling', [
        'transcription.*regulator', 'transcriptional.*regulator',
        'two-component', 'response regulator', 'sensor.*kinase',
        'sigma.*factor', 'regulatory.*protein', 'operon.*regulator',
        'operon.*repressor', 'operon.*activator',
    ]),
    ('Amino acid &\nN metabolism', [
        'amino.*acid', 'aminotransferase', 'deaminase', 'racemase',
        'd-serine', 'd-alanine', 'd-methionine', 'glutamyl',
        'aspartokinase', 'homoserine', 'nitrogen', 'ammonium',
        'urea', 'peptidase', 'protease.*iii',
    ]),
    ('Transport, biosynthesis\n& housekeeping', [
        'microcompartment', 'propanediol',
        'folate', 'thiamine', 'riboflavin', 'dna.*replication', 'dna.*primase',
        'translation', 'trna', 'aminoacyl', 'ribosom',
        'heme.*transport', 'clp.*protease', 'siderophore',
        'pyrimidine', 'beta-oxidation', 'aldehyde', 'lps.*biosynthesis',
        'transport.*system', 'permease', 'abc.*transport', 'mfs.*transport',
        'azoreductase', 'adenylyltransferase', 'kinase',
    ]),
]

P3_CATS = [
    ('Cell cycle, mitosis\n& cytoskeleton', [
        'cell cycle', 'cyclin', 'cdc', 'separase', 'anaphase',
        'kinetochore', 'cohesin', 'mitosis', 'meiosis', 'spindle',
        'tubulin', 'actin', 'cytoskeleton', 'kinesin', 'dynein', 'myosin',
    ]),
    ('Chromatin, transcription\n& RNA processing', [
        'chromatin', 'histone', 'nucleosome', 'mediator', 'spt20',
        'transcription.*initiation', 'transcription.*elongation',
        'rna.*polymerase.*subunit', 'nuclear.*pore', 'nucleoporin',
        'rna.*processing', 'rna.*helicase', 'rna.*splicing', 'spliceosome',
        'ribosome.*biogenesis', 'nucleolar', 'exon.*junction',
        'polyadenylation', 'mrna.*decapping', 'cap-binding',
    ]),
    ('DNA replication\n& repair', [
        'dna.*replication', 'dna.*repair', 'dna.*damage', 'dna.*recombination',
        'dna.*polymerase', 'dna.*helicase', 'dna.*ligase', 'telomere',
        'exonuclease', 'endonuclease', 'topoisomerase', 'checkpoint',
    ]),
    ('Vesicle trafficking\n& autophagy', [
        'vesicle', 'golgi', 'endoplasmic reticulum', 'er membrane',
        'autophagy', 'vacuol', 'lysosome', 'peroxisome', 'peroxin',
        'gpi.*anchor', 'gpi.*mannosyl', 'gpi.*transamidase',
        'exocyst', 'trafficking', 'sorting', 'snare', 'rab.*gtp',
        'clathrin', 'dynamin', 'cargo.*transport',
    ]),
    ('Fungal cell wall &\nsecondary metabolism', [
        'glucan synthase', 'glucanosyl', 'chitin', 'melanin', 'scytalone',
        'polyketide', 'nonribosomal peptide', 'ferricrocin', 'siderophore',
        'sterol', 'ergosterol', 'mevalonate', 'carotenoid',
        'dimethylallyltryptophan', 'aspyridone', 'mevalonyl',
    ]),
    ('Signaling & protein\nturnover', [
        'mapk', 'map.*kinase', 'calmodulin', 'calcium.*kinase',
        'tor.*signaling', 'target of rapamycin', 'ubiquitin', 'sumo',
        'proteasome', 'protease', 'peptidase', 'calpain',
        'phosphatase', 'kinase.*inhibitor', 'f-box', 'cullin',
    ]),
    ('Mitochondrial\nfunction', [
        'mitochondri', 'cytochrome.*oxidase.*subunit', 'atp.*synthase',
        'f-type.*atp', 'complex.*i.*subunit', 'coenzyme.*q',
        'nadh.*dehydrogenase.*ubiquinone',
    ]),
    ('Transcription &\nregulation (bact.)', [
        'transcription.*regulator', 'transcriptional.*regulator',
        'two-component', 'response regulator', 'sensor.*kinase',
        'sigma.*factor', 'anti-sigma', 'dna-binding',
        'transcription.*activator', 'transcription.*repressor',
    ]),
    ('C, N & cofactor\nmetabolism', [
        'urea', 'arginine', 'proline.*utilization', 'polyamine',
        'amino acid', 'aminotransferase', 'deaminase', 'transaminase',
        'sugar', 'mannitol', 'glycerol', 'glycerate', 'propionate',
        'beta-oxidation', 'acyl-coa', 'fatty acid', 'phospholipid',
        'aromatic', 'purine', 'pyrimidine', 'nucleotide', 'nucleoside',
        'folate', 'cobalamin', 'vitamin', 'inositol', 'choline',
        'sulfonate', 'methylotrophy', 'alcohol dehydrogenase',
        'aldehyde.*dehydrogenase', 'carnitine', 'sarcosine',
        'dehydrogenase', 'reductase', 'synthase', 'transferase',
        'oxidase', 'kinase', 'ligase', 'decarboxylase', 'lyase',
        'isomerase', 'mutase', 'epimerase', 'racemase',
    ]),
    ('Transport, membrane\n& other', [
        'transport', 'permease', 'channel', 'porin', 'exporter',
        'importer', 'cation', 'anion', 'phosphate', 'membrane',
        'cell.*adhesion', 'heat.*shock', 'uncharacterized',
        'hypothetical', 'virus', 'mycovirus',
        'repeat-containing', 'domain-containing', 'family.*member',
        'structural.*maintenance', 'chromosome', 'nuclear',
    ]),
]

P4_CATS = [
    ('CAZymes &\npolymer hydrolysis', [
        'cellulase', 'cellobiosidase', 'endoglucanase', 'xylanase',
        'xylosidase', 'glucosidase', 'galactosidase', 'mannosidase',
        'mannanase', 'fucosidase', 'arabinofuranohydrolase', 'pectate',
        'polygalacturonase', 'rhamnogalacturonan', 'glucuronoarabinoxylan',
        'xyloglucan', 'starch.*binding', 'dextranase', 'amylase',
        'pullulanase', 'laminarinase', 'chitinase', 'chitin.*deacetylase',
        'glycoside hydrolase', 'polysaccharide lyase',
    ]),
    ('Sugar transport\n& uptake', [
        'sugar transport', 'sugar.*permease', 'sugar.*abc',
        'galactofuranose.*transport', 'raffinose.*transport', 'melibiose',
        'glucokinase', 'ribulokinase', 'gluconolactonase',
        'pentose.*catab', 'mannonate', 'arabinose isomerase',
        'ribulose.*epimerase', 'cellobiose.*epimerase',
        'pts.*system', 'phosphotransferase.*system',
        'glucose.*dehydrogenase', 'glucose.*oxidase',
    ]),
    ('Aromatic & lignin\ncatabolism', [
        'catechol', 'protocatechuate', 'lignin', 'phenol.*monooxygenase',
        'toluene', 'benzoate', 'vanillate', 'aromatic', 'dioxygenase',
    ]),
    ('Secretion systems\n& motility', [
        'type.*ii.*secretion', 'type.*iii.*secretion', 'type.*vi.*secretion',
        'general secretion pathway', 't2ss', 't3ss', 't6ss',
        'pilus.*assembly', 'cpac', 'cpad', 'cpaf', 'tad.*pilus',
        'adhesin', 'adhesion', 'fibronectin', 'gliding.*motility',
        'outer membrane.*protein.*insertion', 'bamb',
    ]),
    ('Oxidative metabolism\n& respiration', [
        'cytochrome', 'quinol', 'nadh.*quinone', 'oxidoreductase',
        'formate.*dehydrogenase', 'alcohol.*dehydrogenase', 'alcohol.*oxidase',
        'aldehyde.*dehydrogenase', 'dmso', 'nitrate.*reductase',
        'glycolysis', 'pyruvate.*kinase', 'gluconeogenesis',
        'acetate.*kinase', 'propionate', 'lactate.*dehydrogenase',
    ]),
    ('Sulfur & nitrogen\nassimilation', [
        'arylsulfatase', 'sulfatase', 'sulfur', 'arginine', 'asparagine',
        'asparaginase', 'amino acid biosynthesis', 'polyamine',
        'creatinine', 'lysine', 'histidine.*biosynthesis',
        'nitrogen.*fixation', 'nitrogenase', 'nitrate.*transport',
        'nitrite.*transport',
    ]),
    ('Membrane, cell wall\n& hopanoids', [
        'hopanoid', 'menaquinone', 'lipopolysaccharide', 'lps',
        'cell wall', 'cell envelope', 'peptidoglycan', 'murein',
        'fatty acid', 'phospholipid', 'vitamin.*k', 'antibiotic.*resistance',
        'multidrug', 'efflux', 'toxin.*export', 'beta-lactamase',
    ]),
    ('Regulation &\nsignaling', [
        'transcription.*regulator', 'transcriptional.*regulator',
        'two-component', 'response regulator', 'sensor.*kinase',
        'sigma.*factor', 'circadian', 'stress',
    ]),
    ('Amino acid &\nN metabolism', [
        'aminotransferase', 'amino.*acid', 'deaminase', 'racemase',
        'glutamate', 'glutamine', 'asparagine', 'aspartate',
        'alanine', 'glycine.*cleavage', 'serine.*hydroxymethyl',
        'nitrogen.*fixation', 'nitrogenase', 'ammonium',
        'urea', 'peptidase', 'proteinase', 'aminopeptidase',
    ]),
    ('RNA, ribosome &\nnucleotide metabolism', [
        'rrna', 'ribosom', 'rna.*helicase', 'rnase', 'ribonuclease',
        'trna', 'aminoacyl', 'translation', 'elongation.*factor',
        'nucleotid', 'purine', 'pyrimidine', 'selenoprotein',
    ]),
    ('Iron, metal &\nredox defense', [
        'iron', 'ferr', 'siderophore', 'cobalt', 'manganese', 'zinc',
        'copper', 'nickel', 'peroxiredoxin', 'glutaredoxin', 'glutathione',
        'catalase', 'superoxide', 'thioredoxin', 'redox', 'antioxidant',
    ]),
    ('Housekeeping,\ndefense & other', [
        'protease', 'proteasome',
        'dna.*repair', 'dna.*restriction', 'restriction.*enzyme',
        'endonuclease', 'esterase', 'transport', 'permease',
        'glycerol', 'aminosugar', 'polysaccharide', 'glycan',
        'secondary.*metab', 'cytokinin', 'chromosome', 'cytoskeleton',
        'mitochondri', 'uncharacterized', 'hypothetical',
        'dehydrogenase', 'reductase', 'synthase', 'transferase',
        'oxidase', 'kinase', 'ligase', 'carboxylase',
    ]),
]

P5_CATS = [
    ('Sporulation &\ngermination', [
        'sporulat', 'spore.*coat', 'dipicolinate', 'germination',
        'sasp', 'acid-soluble.*spore', 'stage.*ii.*sporulation',
        'cellulase', 'endopeptidase', 'peptidoglycan.*endopeptidase',
        'polyamine', 'tRNA.*modification', 'tRNA.*pseudouridine',
        'vitamin.*transport', 'aldoxime',
    ]),
    ('Methanogenesis &\nanaerobic C1', [
        'methanogen', 'heterodisulfide.*reductase', 'dimethylamine.*corrinoid',
        'dimethylamine.*methyltransferase', 'menaquinone.*reductase',
        'corrinoid.*protein', 'methanol.*corrinoid',
    ]),
    ('Polymer hydrolysis\n& uptake', [
        'cellulase', 'endoglucanase', 'cellobiosidase', 'pectin',
        'polygalacturonan', 'rhamnogalacturonan', 'xylanase',
        'glucosidase', 'galactosidase', 'mannosidase',
    ]),
    ('Secretion, motility\n& cell surface', [
        'pilus', 'secretion', 'flagell', 'chemotaxis', 'adhesin',
        'outer membrane', 'cell wall', 'peptidoglycan',
        'lps', 'lipopolysaccharide', 'efflux', 'multidrug',
    ]),
    ('Regulation &\nsignaling', [
        'transcription.*regulator', 'transcriptional.*regulator',
        'two-component', 'response regulator', 'sensor.*kinase',
        'sigma.*factor',
    ]),
    ('Amino acid, N &\ncofactor metabolism', [
        'amino.*acid', 'aminotransferase', 'deaminase', 'glutamate',
        'glutamine', 'nitrogen', 'nitrate', 'nitrite', 'ammonium',
        'urea', 'siderophore', 'iron', 'cobalt', 'heme', 'porphyrin',
        'cobalamin', 'folate', 'thiamine', 'riboflavin', 'pyridoxal',
        'sulfur', 'sulfate', 'cysteine',
    ]),
    ('Transport &\nhousekeeping', [
        'transport', 'permease', 'abc.*binding', 'mfs.*transporter',
        'ribosom', 'trna', 'translation', 'dna.*replication', 'dna.*repair',
        'protease', 'peptidase', 'chaperone',
        'dehydrogenase', 'reductase', 'synthase', 'transferase',
        'oxidase', 'kinase', 'ligase',
    ]),
]

# ============================================================================
# K=3 subcategory definitions — unified for three programs
# ============================================================================

P1_K3_CATS = [
    ('TCA cycle &\nglycolysis', [
        'pyruvate kinase', 'pyruvate dehydrogenase', 'dihydrolipoyl',
        'citrate synthase', 'aconitase', 'aconitate', 'isocitrate dehydrogenase',
        'oxoglutarate dehydrogenase', '2-oxoglutarate', 'succinate dehydrogenase',
        'succinyl-coa', 'fumarase', 'fumarate hydratase', 'malate dehydrogenase',
        'phosphoglycerate', 'enolase', 'glucose-6-phosphate isomerase',
        'glyceraldehyde-3-phosphate dehydrogenase', 'transketolase', 'transaldolase',
        'fructose-bisphosphate aldolase', 'phosphofructokinase',
        'pentose phosphate',
    ]),
    ('Fermentation &\noverflow metabolism', [
        'lactate dehydrogenase', 'butanol dehydrogenase', 'alcohol dehydrogenase',
        'formate dehydrogenase', 'formate hydrogenlyase', 'hydrogenase',
        'acetaldehyde dehydrogenase', 'acetate kinase', 'phosphotransacetylase',
        'pyruvate formate-lyase', 'glycyl radical', 'ethanolamine',
        'propanediol dehydratase', 'microcompartment',
        'methylglyoxal', 'acetoin', 'citrate lyase',
        'glyoxylate', 'malate synthase', 'isocitrate.*lyase', 'methylcitrate',
    ]),
    ('Chaperone &\nstress response', [
        'chaperone', 'chaperonin', 'dnaj', 'dnak', 'grpe', 'groel', 'groes',
        'clp.*protease', 'clpb', 'clpc', 'lon protease', 'ftsh',
        'heat.*shock', 'heat.*inducible', 'htpx', 'hsp20', 'ibpa',
        'cold.*shock', 'phage.*shock', 'stress',
    ]),
    ('Energy conservation\n& respiration', [
        'atp synthase', 'cytochrome', 'quinol', 'nadh.*quinone', 'nadh.*ubiquinone',
        'ferredoxin', 'electron.*transfer', 'rnf complex',
        'thioredoxin', 'glutaredoxin', 'peroxiredoxin', 'catalase',
        'superoxide', 'fe.*s.*cluster', 'fe.*s.*biogenesis',
    ]),
    ('Regulation &\nsignaling', [
        'transcription.*regulator', 'transcriptional.*regulator',
        'two-component', 'response regulator', 'sensor.*kinase',
        'sigma.*factor', 'anti-sigma', 'adenylate cyclase',
        'diguanylate cyclase',
    ]),
    ('Transport &\nPTS', [
        'sugar transport', 'pts.*system', 'phosphotransferase system',
        'abc transport', 'mfs transporter', 'permease', 'solute carrier',
        'porin', 'outer membrane protein', 'tonb',
        'transport.*system',
    ]),
    ('Motility, cell surface\n& secretion', [
        'flagell', 'chemotaxis', 'pilus', 'adhesin', 'adhesion',
        'lps', 'lipopolysaccharide', 'cell wall', 'peptidoglycan', 'murein',
        'multidrug', 'efflux', 'secretion',
        'preprotein translocase',
    ]),
    ('Nitrogen\nmetabolism', [
        'nitrogen fix', 'nitrogenase', 'nif', 'nitrate', 'nitrite',
        'ammonium', 'glutamine synthetase', 'glutamate synthase',
        'urea', 'urease',
    ]),
    ('Amino acid\nmetabolism', [
        'aminotransferase', 'amino.*acid', 'deaminase', 'racemase',
        'glutamate.*dehydrogenase', 'aspartate', 'arginine',
        'proline', 'lysine', 'histidin', 'tryptophan',
        'branched-chain',
    ]),
    ('Lipid & fatty acid\nmetabolism', [
        'acyl-coa dehydrogenase', 'acetyl-coa acyltransferase',
        'enoyl-coa', 'fatty acid', 'phospholipid', 'lipid',
        'acyl.*carrier.*protein',
    ]),
    ('Cofactor\nbiosynthesis', [
        'biotin', 'cobalamin', 'folate', 'thiamine', 'riboflavin',
        'nad biosynthesis', 'heme', 'porphyrin', 'menaquinone',
        'pyridoxal', 'coenzyme',
    ]),
    ('DNA, RNA &\ntranslation', [
        'ribosom', 'rrna', 'trna', 'translation', 'elongation factor',
        'dna.*replication', 'dna.*repair', 'dna.*polymerase', 'dna.*helicase',
        'dna.*gyrase', 'topoisomerase', 'cell division', 'fts',
        'rna.*polymerase', 'rnase', 'ribonuclease',
    ]),
    ('Iron & metal\nhomeostasis', [
        'iron', 'siderophore', 'ferrous', 'ferric', 'zinc', 'copper',
        'cobalt', 'manganese',
    ]),
]

P2_K3_CATS = [
    ('Eukaryotic cell cycle\n& chromatin', [
        'kinetochore', 'histone', 'chromatin', 'nucleosome', 'nuclear pore',
        'nucleoporin', 'spliceosome', 'cyclin', 'separase', 'anaphase',
        'cohesin', 'mediator', 'spindle', 'centromere', 'telomere',
        'cdc', 'meiosis',
    ]),
    ('Vesicle trafficking\n& autophagy', [
        'vesicle', 'golgi', 'endoplasmic reticulum', 'er membrane',
        'autophagy', 'vacuol', 'lysosome', 'peroxisome', 'peroxin',
        'gpi.*anchor', 'exocyst', 'trafficking', 'sorting',
        'clathrin', 'dynamin', 'cargo.*transport',
    ]),
    ('Fungal cell wall &\nsecondary metabolism', [
        'glucan synthase', 'glucanosyl', 'chitin', 'melanin', 'scytalone',
        'polyketide', 'nonribosomal peptide', 'ferricrocin',
        'sterol', 'ergosterol', 'mevalonate',
        'fatty acid synthase.*fungi',
    ]),
    ('Mitochondrial\nfunction', [
        'mitochondri', 'cytochrome.*oxidase.*subunit', 'f-type.*atp',
        'complex.*i.*subunit', 'coenzyme.*q',
    ]),
    ('Signaling &\nprotein turnover', [
        'mapk', 'map.*kinase', 'calmodulin', 'target of rapamycin',
        'ubiquitin', 'sumo', 'proteasome', 'protease', 'peptidase',
        'phosphatase', 'kinase.*inhibitor', 'f-box',
    ]),
    ('Stress & redox\ndefense', [
        'thioredoxin', 'glutaredoxin', 'peroxiredoxin', 'catalase',
        'superoxide', 'chaperone', 'dnak', 'clp.*protease', 'heat.*shock',
        'fe.*s.*cluster', 'sufb',
    ]),
    ('Transcription &\nregulation', [
        'transcription.*regulator', 'transcriptional.*regulator',
        'two-component', 'response regulator', 'sensor.*kinase',
        'sigma.*factor', 'dna-binding',
    ]),
    ('Carbon, nitrogen &\ncofactor metabolism', [
        'dehydrogenase', 'reductase', 'synthase', 'transferase',
        'oxidase', 'kinase', 'ligase', 'decarboxylase', 'lyase',
        'amino.*acid', 'aminotransferase', 'glutamate', 'arginine',
        'proline', 'urea', 'polyamine',
        'sugar', 'glycerol', 'propionate', 'aromatic',
        'purine', 'pyrimidine', 'nucleotide', 'folate', 'cobalamin',
    ]),
    ('Transport &\nmembrane', [
        'transport', 'permease', 'channel', 'porin', 'exporter',
        'importer', 'membrane', 'abc',
    ]),
]

P3_K3_CATS = [
    ('CAZymes &\npolymer hydrolysis', [
        'cellulase', 'cellobiosidase', 'endoglucanase', 'xylanase',
        'xylosidase', 'glucosidase', 'galactosidase', 'mannosidase',
        'fucosidase', 'arabinofuranohydrolase', 'pectate',
        'polysaccharide lyase', 'glycoside hydrolase',
        'starch.*binding', 'susc', 'dextranase',
    ]),
    ('Sugar transport\n& uptake', [
        'sugar transport', 'sugar.*permease', 'pts.*system',
        'phosphotransferase system', 'glucokinase',
        'glucose.*dehydrogenase',
    ]),
    ('Surface attachment\n& secretion', [
        'pilus', 'type.*iv.*pilus', 'type.*ii.*secretion', 'type.*vi.*secretion',
        'general secretion pathway', 'tonb', 'adhesin', 'fibronectin',
        'gliding.*motility', 'twitching', 'outer membrane',
    ]),
    ('Amino acid biosynthesis\n& N assimilation', [
        'glutamine synthetase', 'glutamate synthase', 'asparagine synthase',
        'aspartate aminotransferase', 'chorismate', 'prephenate',
        'diaminopimelate', 'histidinol', 'anthranilate',
        'nitrogen fix', 'nitrogenase', 'nif', 'nitrate', 'nitrite',
        'ammonium',
    ]),
    ('Amino acid catabolism\n& proteolysis', [
        'aminotransferase', 'peptidase', 'protease', 'proteinase',
        'deaminase', 'racemase', 'amino.*acid.*transport',
    ]),
    ('Aromatic & lignin\ncatabolism', [
        'catechol', 'protocatechuate', 'lignin', 'phenol.*monooxygenase',
        'toluene', 'benzoate', 'aromatic', 'dioxygenase',
    ]),
    ('Membrane, cell wall\n& hopanoids', [
        'hopanoid', 'menaquinone', 'lipopolysaccharide', 'lps',
        'cell wall', 'peptidoglycan', 'murein', 'fatty acid',
        'phospholipid', 'multidrug', 'efflux',
    ]),
    ('Regulation &\nsignaling', [
        'transcription.*regulator', 'transcriptional.*regulator',
        'two-component', 'response regulator', 'sensor.*kinase',
        'sigma.*factor',
    ]),
    ('Energy &\nrespiration', [
        'cytochrome', 'quinol', 'nadh', 'ferredoxin', 'electron.*transfer',
        'atp synthase', 'formate dehydrogenase',
    ]),
    ('DNA, RNA &\ntranslation', [
        'ribosom', 'trna', 'translation', 'dna.*repair', 'dna.*replication',
        'dna.*polymerase', 'rna.*helicase', 'rnase',
    ]),
]

SUBCAT_DEFS = {
    'P_EARLY_MONO': P1_K3_CATS,
    'P_LATE_MONO': P2_K3_CATS,
    'P_HEMI': P3_K3_CATS,
}

# ============================================================================
# Load and classify
# ============================================================================
print("Loading k-means program assignments...")
df = pd.read_csv(os.path.join(DERIVED_DIR, 'kmeans_program_assignments.tsv'), sep='\t')

mono_cols = [c for c in df.columns if c.startswith('MONOMER_')]
hemi_cols = [c for c in df.columns if c.startswith('HEMICELLULOSE_')]
all_data_cols = mono_cols + hemi_cols

mono_tp_labels = ['2h', '8h', '1d', '3d', '7d', '14d']
hemi_tp_labels = ['2h', '8h', '1d', '3d', '7d', '14d', '28d']
all_tp_labels = mono_tp_labels + hemi_tp_labels

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

print("\nSubcategory classification:")
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
out_tsv = os.path.join(TABLES_DIR, 'TableS_program_subcategory_assignments_v6.tsv')
df_out = df.copy()
df_out['subcategory_flat'] = df_out.subcategory.str.replace('\n', ' ')
df_out.to_csv(out_tsv, sep='\t', index=False)
print(f"\nSaved: {out_tsv}")

# ============================================================================
# Build summary: mean LFC per subcategory
# ============================================================================
subcat_rows = []
for prog in PROG_ORDER:
    cats = SUBCAT_DEFS[prog]
    sub = df[df.mnemonic == prog]
    for cat_label, _ in cats:
        n = (sub.subcategory == cat_label).sum()
        if n > 0:
            subcat_rows.append((prog, cat_label, n))
    n_other = (sub.subcategory == 'Other/diverse').sum()
    if n_other > 0:
        subcat_rows.append((prog, 'Other/diverse', n_other))

summary = []
for prog, cat, n in subcat_rows:
    mask = (df.mnemonic == prog) & (df.subcategory == cat)
    means = df.loc[mask, all_data_cols].mean(axis=0).values
    summary.append({
        'program': prog, 'subcategory': cat, 'n_kos': n,
        **{col: val for col, val in zip(all_data_cols, means)}
    })

df_summary = pd.DataFrame(summary)
n_rows = len(df_summary)
print(f"\nTotal subcategory rows: {n_rows}")

# ============================================================================
# FIGURE
# ============================================================================
print("Generating subprogram heatmap...")

n_mono = len(mono_cols)
n_hemi = len(hemi_cols)
n_tp = n_mono + n_hemi

fig_w = 180 * MM
fig_h = max(n_rows * 7.5 * MM + 30 * MM, 120 * MM)

fig = plt.figure(figsize=(fig_w, fig_h))

plot_top = 0.93
plot_bot = 0.06
prog_left = 0.00
prog_width = 0.04
heat_left = 0.33
heat_right = 0.90
cbar_left = 0.915
cbar_right = 0.94

ax_prog = fig.add_axes([prog_left, plot_bot, prog_width, plot_top - plot_bot])
ax_prog.set_xlim(0, 1)
ax_prog.set_ylim(n_rows - 0.5, -0.5)
ax_prog.axis('off')

ax = fig.add_axes([heat_left, plot_bot, heat_right - heat_left, plot_top - plot_bot])
cax = fig.add_axes([cbar_left, plot_bot, cbar_right - cbar_left, plot_top - plot_bot])

heat = df_summary[all_data_cols].values
vmax = min(max(np.nanmax(np.abs(heat)), 1.0), 5.0)
norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

im = ax.imshow(heat, aspect='auto', cmap='RdBu_r', norm=norm, interpolation='nearest')

ax.axvline(n_mono - 0.5, color='black', linewidth=1.5, zorder=5)

ax.text(n_mono / 2 - 0.5, -1.2, 'Monomers\n(glucose + xylose)',
        fontsize=7, fontfamily=FF, ha='center', va='bottom',
        fontweight='bold', color='#6B7C2E')
ax.text(n_mono + n_hemi / 2 - 0.5, -1.2, 'Hemicellulose',
        fontsize=7, fontfamily=FF, ha='center', va='bottom',
        fontweight='bold', color='#7B3F6E')

ax.set_xticks(range(n_tp))
ax.set_xticklabels(all_tp_labels, fontsize=7, fontfamily=FF, rotation=0)
ax.tick_params(axis='x', length=2, pad=2)

row_labels = [f"{r['subcategory'].replace(chr(10), ' ')}  (n={r['n_kos']})"
              for _, r in df_summary.iterrows()]
ax.set_yticks(range(n_rows))
ax.set_yticklabels(row_labels, fontsize=6, fontfamily=FF)
ax.tick_params(axis='y', length=0, pad=4)

for i in range(n_rows):
    for j in range(n_tp):
        val = heat[i, j]
        if np.isnan(val):
            continue
        txt_color = 'white' if abs(val) > vmax * 0.55 else 'black'
        ax.text(j, i, f'{val:.1f}', ha='center', va='center',
                fontsize=4.0, fontfamily=FF, color=txt_color, fontweight='bold')

prog_boundaries = []
for prog in PROG_ORDER:
    prog_rows = [i for i, (p, _, _) in enumerate(subcat_rows) if p == prog]
    if not prog_rows:
        continue
    first, last = prog_rows[0], prog_rows[-1]

    for idx in prog_rows:
        if idx < len(ax.get_yticklabels()):
            ax.get_yticklabels()[idx].set_color(PROG_COLORS[prog])

    if first > 0:
        ax.axhline(first - 0.5, color='white', linewidth=2.5, zorder=4)
        prog_boundaries.append(first - 0.5)

    mid = (first + last) / 2
    n_prog = len(df[df.mnemonic == prog])
    ax_prog.text(0.5, mid, f'{PROG_NAMES[prog]}  (n={n_prog})',
                 fontsize=6, fontfamily=FF, fontweight='bold',
                 color=PROG_COLORS[prog], ha='center', va='center',
                 rotation=90)
    ax_prog.plot([0.85, 0.85], [first - 0.3, last + 0.3],
                 color=PROG_COLORS[prog], linewidth=2.0, solid_capstyle='round',
                 clip_on=False)

ax.set_xlim(-0.5, n_tp - 0.5)
ax.set_ylim(n_rows - 0.5, -0.5)
for sp in ax.spines.values():
    sp.set_visible(False)

cb = plt.colorbar(im, cax=cax)
cb.set_label('Mean log$_2$FC vs control', fontsize=7, fontfamily=FF)
cb.ax.tick_params(labelsize=6)

fig.suptitle('Transcriptional subprograms across substrate treatments',
             fontsize=9, fontfamily=FF, fontweight='bold', y=0.97)
fig.text(0.62, 0.94,
         f'Mean VST LFC across KOs within each subcategory (n={len(df)} interaction-significant KOs)',
         fontsize=5.5, fontfamily=FF, ha='center', color='#666666')

for ext in ['pdf', 'png']:
    fig.savefig(os.path.join(FIGURES_DIR, f'Figure_subprograms_heatmap.{ext}'),
                format=ext, dpi=600, bbox_inches='tight')
plt.close(fig)
print("Saved Figure_subprograms_heatmap.pdf/.png")
