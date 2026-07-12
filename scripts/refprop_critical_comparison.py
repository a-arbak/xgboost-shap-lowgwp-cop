"""
Compares the linear mass-fraction-weighted mixing rule (Eq. 1 in manuscript.tex)
against REFPROP 10.0's rigorous multi-fluid Helmholtz-energy mixture model
for the critical point (Tc, Pc) of all zeotropic/azeotropic blends in the dataset.

Requested by Reviewer #2 (JIJR-D-26-00648): quantify how far the first-order
linear approximation deviates from the true mixture critical point for
zeotropic blends with large temperature glide.
"""
import os
from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary

RP_PATH = os.environ.get('RPPREFIX', r'C:\Program Files (x86)\REFPROP')
OUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       os.pardir, 'data', 'refprop_critical_comparison.csv')

RP = REFPROPFunctionLibrary(os.path.join(RP_PATH, 'REFPRP64.DLL'))
RP.SETPATHdll(RP_PATH)

# (label, REFPROP fluid string, mass fractions, linear-rule Tc_mix [K], linear-rule Pc_mix [MPa])
# Linear-rule values taken directly from Table "tbl:descriptors" in manuscript.tex
BLENDS = [
    ('R-513A', 'R134A;R1234YF',        [0.44, 0.56],       370.65, 3.68),
    ('R-450A', 'R134A;R1234ZEE',       [0.42, 0.58],       379.03, 3.81),
    ('Tern-1', 'R134A;R1234YF;R1234ZEE', [0.492, 0.338, 0.170], 373.47, 3.76),
    ('R-515B', 'R1234ZEE;R227EA',      [0.911, 0.089],     381.84, 3.57),
    ('R-410A', 'R32;R125',             [0.50, 0.50],       345.22, 4.70),
    ('R-454B', 'R32;R1234YF',          [0.689, 0.311],     356.42, 5.04),
    ('R-452B', 'R32;R125;R1234YF',     [0.67, 0.07, 0.26], 354.73, 5.01),
]


def main():
    print(f"REFPROP version: {RP.RPVersion()}\n")
    rows = []
    header = (f"{'Fluid':8s} {'Tc_lin[K]':>10s} {'Tc_REFPROP[K]':>14s} {'dTc[K]':>8s} "
              f"{'Pc_lin[MPa]':>12s} {'Pc_REFPROP[MPa]':>16s} {'dPc[MPa]':>9s} {'dPc[%]':>7s}  status")
    print(header)
    print('-' * len(header))

    for label, hfld, z, tc_lin, pc_lin in BLENDS:
        r = RP.REFPROPdll(hfld, '', 'TC;PC', 21, 1, 0, 0, 0, z)
        tc_rp = r.Output[0]
        pc_rp = r.Output[1] / 1e6  # Pa -> MPa
        dTc = tc_rp - tc_lin
        dPc = pc_rp - pc_lin
        dPc_pct = 100 * dPc / pc_lin
        status = 'estimated (Type I mix)' if r.ierr != 0 else 'exact'
        print(f"{label:8s} {tc_lin:10.2f} {tc_rp:14.2f} {dTc:8.2f} "
              f"{pc_lin:12.2f} {pc_rp:16.3f} {dPc:9.3f} {dPc_pct:7.2f}  {status}")
        rows.append(dict(fluid=label, Tc_linear_K=tc_lin, Tc_REFPROP_K=round(tc_rp, 2),
                          dTc_K=round(dTc, 2), Pc_linear_MPa=pc_lin,
                          Pc_REFPROP_MPa=round(pc_rp, 3), dPc_MPa=round(dPc, 3),
                          dPc_pct=round(dPc_pct, 2), refprop_status=status,
                          refprop_ierr=r.ierr, refprop_herr=r.herr.strip()))

    import csv
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {OUT_CSV}")


if __name__ == '__main__':
    main()
