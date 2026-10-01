"""Check alignment coordinates at the FScanR-to-window boundary."""
from Bio.Seq import Seq
import pandas as pd
import pytest

from FScanpy import fscanr, extract_prf_regions
from FScanpy.utils import extract_window_sequences


def test_different_peptide_sites_are_not_deduplicated_by_dna_coordinates():
    rows = []
    for name, peptide_start in [('gene_a', 1), ('gene_b', 18)]:
        rows.extend([
            [name,'protein',99,33,0,0,1,99,peptide_start,peptide_start+32,1e-20,100,1,0],
            [name,'protein',99,30,0,0,101,190,peptide_start+33,peptide_start+62,1e-20,100,2,0],
        ])
    result = fscanr(pd.DataFrame(rows))
    assert result.DNA_seqid.tolist() == ['gene_a','gene_b']
    assert result.Pep_FS_start.tolist() == [34,51]


def test_peptide_gap_cutoff_uses_amino_acids_not_nucleotides():
    rows = [
        ['gene','protein',99,33,0,0,1,99,1,33,1e-20,100,1,0],
        ['gene','protein',99,30,0,0,110,199,37,66,1e-20,100,2,0],
    ]
    assert fscanr(pd.DataFrame(rows),frameDist_cutoff=10).empty


@pytest.mark.parametrize('strand', ['+','-'])
def test_explicit_zero_based_input_uses_the_same_physical_position(tmp_path, strand):
    sequence = 'ACGT' * 225
    fasta = tmp_path / 'sequence.fasta'
    fasta.write_text('>gene\n'+sequence+'\n')
    sites = pd.DataFrame({'DNA_seqid':['gene'], 'FS_start':[603], 'FS_end':[604],
                          'Strand':[strand], 'FS_type':[1]})
    expected = extract_prf_regions(str(fasta),sites)
    zero_based = sites.assign(FS_start=sites.FS_start-1, FS_end=sites.FS_end-1)
    actual = extract_prf_regions(str(fasta),zero_based,coordinate_base=0)
    assert expected['399bp'].tolist() == actual['399bp'].tolist()


@pytest.mark.parametrize('strand, position', [('+',603), ('-',734)])
def test_blastx_position_is_converted_to_coding_strand_before_window_extraction(tmp_path, strand, position):
    sequence = 'ACGT' * 225
    fasta = tmp_path / 'sequence.fasta'
    fasta.write_text('>gene\n'+sequence+'\n')
    sites = pd.DataFrame({'DNA_seqid':['gene'], 'FS_start':[position], 'FS_end':[position+1],
                          'Strand':[strand], 'FS_type':[1]})
    result = extract_prf_regions(str(fasta),sites)
    oriented = sequence if strand=='+' else str(Seq(sequence).reverse_complement())
    converted = position-1 if strand=='+' else len(sequence)-position
    assert result['399bp'].iloc[0] == extract_window_sequences(oriented,converted)[1]
    assert result.FS_start.iloc[0] == position
