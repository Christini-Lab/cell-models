import math
from typing import List
from cell_models.cell_model import CellModel

import numpy as np
from scipy import integrate

from cell_models import protocols
from cell_models import trace
from math import log, exp


class OharaRudyModel(CellModel):
    """An implementation of the Kernik model by Kernik et al.

    Attributes:
        default_parameters: A dict containing tunable parameters along with
            their default values as specified in Kernik et al.
        updated_parameters: A dict containing all parameters that are being
            tuned.
    """

    cm_farad = 1.0

    parameters = np.array([0, 1.8, 5.4, 140.0, 96485.0,\
        8314.0, 310.0, 2, 1,\
        1, 0.01, 0.0011, -80.0, 1e+17, 500.0, 1, 10.0, 0.05, 0.0015, 0.15,\
        0.05, 0.00068, 1.124, 0.047, 0.0087, 0.00087, 1.0, 0.05, 10.0,\
        0.00238, 0.8, 0.0005, 0.07, 0.01833, 0.99, 75.0, 82.9, 6.086, 39.57,\
        9.871, 6.765, 8.552, 11.64, 34.77, 77.42, 5.955, 0.0,\
        0.019957499999999975, 200.0, 0.0008, 0.00015, 12.5, 5000.0,\
        1500000.0, 15.0, 5.0, 88.12, 0.167, 0.5224, 60000.0, 60000.0, 5000.0,\
        1e-07, 1.698e-07, 0.5, 0.3582, 1.698e-07, 9.073, 27.78, 224.0, 292.0,\
        0.05, 9.8, 30.0, -0.155, 4.2, 182.4, 949.5, 39.4, 687.2, 79300.0,\
        1899.0, 40.0, 639.0, 3.75e-10, 0.02, 0.002, 0.0001007, 1000.0,\
        0.0264, 0.0007868, 4.986e-06, 5.455e-06, 0.001214, 0.005509,\
        1.854e-05, 0.001416, 0.4492, 0.3181, 0.149, 0.01241, 0.3226,\
        0.008978, 4.631e-05, 1.535e-08, -0.004226, -0.1688, 0.008516,\
        7.771e-09, -0.04641, -0.02877, 0.008595, 3.613e-08, 0.004668, 0.1725,\
        -0.0006575, -0.02215, 0.04658545454545456, 0, 0.0, 0.0, 37, 1.0, 1,\
        1, 4.843, 4.942, 4.23, 4.156, 4.962, 4.22, 3.769, 1.459, 5, 4.663,\
        2.412, 5.568, 5, 5.682, 0.006358000000000001, 817.3,\
        0.3239783999999998, 0.003, 2.5e-08, 0.0005, 0.0005, 1.0, 4.75, 1.0],\
        dtype=np.float_)



    def __init__(self, default_parameters=None,
                 updated_parameters=None,
                 no_ion_selective_dict=None,
                 default_time_unit='ms', 
                 default_voltage_unit='mV',
                 default_voltage_position=40,
                 concentration_indices={'Nai': 41, 'nass': 42, 'ki': 43,
                                        'kss': 44, 'cai': 45, 'cass': 46, 
                                        'cansr': 47, 'cajsr': 48}
                 ):

        y_initial = np.array([0.0125840447, 0.007344121102, 0.6981071913,\
            0.6980895801, 0.6979908432, 0.4549485525, 0.6979245865,\
            0.0001882617273, 0.5008548855, 0.2693065357, 0.001001097687,\
            0.9995541745, 0.5865061736, 0.0005100862934, 0.9995541823,\
            0.6393399482, 2.34e-09, 0.9999999909, 0.9102412777, 0.9999999909,\
            0.9998046777, 0.9999738312, 0.9999999909, 0.9999999909,\
            0.002749414044, 0.999637, 6.83208e-05, 1.80145e-08, 8.26619e-05,\
            0.00015551, 5.67623e-05, 0, 0, 0, 0, 0.2707758025, 0.0001928503426,\
            0.9967597594, 2.5e-07, 3.12e-07, -88.00190465, 7.268004498,\
            7.268089977, 144.6555918, 144.6555651, 8.6e-05, 8.49e-05,\
            1.619574538, 1.571234014], dtype=np.float_)


        
        super().__init__(concentration_indices, 
                         y_initial, default_parameters,
                         updated_parameters,
                         no_ion_selective_dict,
                         default_time_unit,
                         default_voltage_unit,
                         default_voltage_position=default_voltage_position)

    def action_potential_diff_eq(self, t, y):
        """
        Compute the right hand side of the ohara_rudy_cipa_v1_2017 ODE
        """
        # Assign states
        assert(len(y) == 49)
        CaMKt, m, hf, hs, j, hsp, jp, mL, hL, hLp, a, iF, iS, ap, iFp, iSp, d,\
            ff, fs, fcaf, fcas, jca, ffp, fcafp, nca, IC1, IC2, C1, C2, O, IO,\
            IObound, Obound, Cbound, D, xs1, xs2, xk1, Jrelnp, Jrelp, v, nai,\
            nass, ki, kss, cai, cass, cansr, cajsr = y

        # Assign parameters
        celltype, cao, ko, nao, F, R, T, zca, zk, zna, L, rad,\
        i_Stim_Amplitude, i_Stim_End, i_Stim_Period, i_Stim_PulseDuration,\
        i_Stim_Start, CaMKo, KmCaM, KmCaMK, aCaMK, bCaMK, BSLmax, BSRmax, KmBSL,\
        KmBSR, cm, cmdnmax_b, csqnmax, kmcmdn, kmcsqn, kmtrpn, trpnmax, PKNa,\
        Ahf, GNa, hssV1, hssV2, mssV1, mssV2, mtD1, mtD2, mtV1, mtV2, mtV3,\
        mtV4, shift_INa_inact, GNaL_b, thL, Gncx_b, KmCaAct, kasymm, kcaoff,\
        kcaon, kna1, kna2, kna3, qca, qna, wca, wna, wnaca, H, Khp, Kki, Kko,\
        Kmgatp, Knai0, Knao0, Knap, Kxkur, MgADP, MgATP, Pnak_b, delta, eP,\
        k1m, k1p, k2m, k2p, k3m, k3p, k4m, k4p, PNab, Gto_b, Kmn, PCa_b, k2n,\
        A1, A11, A2, A21, A3, A31, A4, A41, A51, A52, A53, A61, A62, A63,\
        B1, B11, B2, B21, B3, B31, B4, B41, B51, B52, B53, B61, B62, B63,\
        GKr_b, Kmax, Kt, Ku, Temp, Vhalf, halfmax, n, q1, q11, q2, q21, q3,\
        q31, q4, q41, q51, q52, q53, q61, q62, q63, GKs_b, txs1_max, GK1_b,\
        GKb_b, PCab, GpCa, KmCap, Jrel_scaling_factor, bt, Jup_b = self.parameters

        # Init return args
        values = np.zeros((49,), dtype=np.float_)

        # Expressions for the Ohara rudy cipa v1 2017 component
        frt = F/(R*T)
        ffrt = F*frt
        vfrt = frt*v

        # Expressions for the Cell geometry component
        vcell = 3140.0*L*(rad*rad)
        Ageo = 6.28*(rad*rad) + 6.28*L*rad
        Acap = 2.0*Ageo
        vmyo = 0.68*vcell
        vnsr = 0.0552*vcell
        vjsr = 0.0048*vcell
        vss = 0.02*vcell

        # Expressions for the CaMK component
        CaMKb = CaMKo*(1.0 - CaMKt)/(1.0 + KmCaM/cass)
        CaMKa = CaMKb + CaMKt
        values[0] = -bCaMK*CaMKt + aCaMK*(CaMKb + CaMKt)*CaMKb

        # Expressions for the Reversal potentials component
        ENa = R*T*math.log(nao/nai)/F
        EK = R*T*math.log(ko/ki)/F
        EKs = R*T*math.log((ko + PKNa*nao)/(PKNa*nai + ki))/F

        # Expressions for the INa component
        mss = 1.0/(1.0 + math.exp((-mssV1 - v)/mssV2))
        tm = 1.0/(mtD1*math.exp((mtV1 + v)/mtV2) + mtD2*math.exp((-mtV3 - v)/mtV4))
        values[1] = (-m + mss)/tm
        hss = 1.0/(1.0 + math.exp((hssV1 - shift_INa_inact + v)/hssV2))
        thf =\
            1.0/(1.183856958289087e-05*math.exp(0.15910898965791567*shift_INa_inact\
            - 0.15910898965791567*v) +\
            6.305549185817275*math.exp(0.0493339911198816*v -\
            0.0493339911198816*shift_INa_inact))
        ths =\
            1.0/(0.005164670235381792*math.exp(0.035650623885918005*shift_INa_inact\
            - 0.035650623885918005*v) +\
            0.36987619372096325*math.exp(0.017649135192375574*v -\
            0.017649135192375574*shift_INa_inact))
        Ahs = 1.0 - Ahf
        values[2] = (-hf + hss)/thf
        values[3] = (-hs + hss)/ths
        h = Ahf*hf + Ahs*hs
        jss = hss
        tj = 2.038 + 1.0/(0.3131936394738773*math.exp(0.02600780234070221*v -\
            0.02600780234070221*shift_INa_inact) +\
            1.1315282095590072e-07*math.exp(0.12075836251660427*shift_INa_inact -\
            0.12075836251660427*v))
        values[4] = (-j + jss)/tj
        hssp = 1.0/(1.0 + 2281075.816697194*math.exp(0.1643115346697338*v -\
            0.1643115346697338*shift_INa_inact))
        thsp = 3.0*ths
        values[5] = (-hsp + hssp)/thsp
        hp = Ahf*hf + Ahs*hsp
        tjp = 1.46*tj
        values[6] = (-jp + jss)/tjp
        fINap = 1.0/(1.0 + KmCaMK/CaMKa)
        INa = GNa*math.pow(m, 3.0)*(-ENa + v)*((1.0 - fINap)*h*j + fINap*hp*jp)

        # Expressions for the L component
        mLss = 1.0/(1.0 + 0.000291579585635531*math.exp(-0.18996960486322187*v))
        tmL = tm
        values[7] = (-mL + mLss)/tmL
        hLss = 1.0/(1.0 + 120578.15595522427*math.exp(0.13354700854700854*v))
        values[8] = (-hL + hLss)/thL
        hLssp = 1.0/(1.0 + 275969.2903869871*math.exp(0.13354700854700854*v))
        thLp = 3.0*thL
        values[9] = (-hLp + hLssp)/thLp
        GNaL = (0.6*GNaL_b if celltype == 1.0 else GNaL_b)
        fINaLp = 1.0/(1.0 + KmCaMK/CaMKa)
        INaL = (-ENa + v)*((1.0 - fINaLp)*hL + fINaLp*hLp)*GNaL*mL

        # Expressions for the Ito component
        ass = 1.0/(1.0 + 2.6316508161673635*math.exp(-0.06747638326585695*v))
        ta = 1.0515/(1.0/(1.2089 +\
            2.2621017070578837*math.exp(-0.03403513787634354*v)) + 3.5/(1.0 +\
            30.069572727397507*math.exp(0.03403513787634354*v)))
        values[10] = (-a + ass)/ta
        iss = 1.0/(1.0 + 2194.970764538301*math.exp(0.17510068289266328*v))
        delta_epi = (1.0 - 0.95/(1.0 + 1202604.2841647768*math.exp(0.2*v)) if\
            celltype == 1.0 else 1.0)
        tiF_b = 4.562 + 1.0/(0.14468698421272827*math.exp(-0.01*v) +\
            1.6300896349780942*math.exp(0.06027727546714889*v))
        tiS_b = 23.62 +\
            1.0/(0.00027617763953377436*math.exp(-0.01693480101608806*v) +\
            0.024208962804604526*math.exp(0.12377769525931426*v))
        tiF = delta_epi*tiF_b
        tiS = delta_epi*tiS_b
        AiF = 1.0/(1.0 + 0.24348537187522867*math.exp(0.006613756613756614*v))
        AiS = 1.0 - AiF
        values[11] = (-iF + iss)/tiF
        values[12] = (-iS + iss)/tiS
        i = AiF*iF + AiS*iS
        assp = 1.0/(1.0 + 5.167428462230666*math.exp(-0.06747638326585695*v))
        values[13] = (-ap + assp)/ta
        dti_develop = 1.354 +\
            0.0001/(2.6591269045230603e-05*math.exp(0.06293266205160478*v) +\
            4.5541779737128264e+24*math.exp(-4.642525533890436*v))
        dti_recover = 1.0 - 0.5/(1.0 + 33.11545195869231*math.exp(0.05*v))
        tiFp = dti_develop*dti_recover*tiF
        tiSp = dti_develop*dti_recover*tiS
        values[14] = (-iFp + iss)/tiFp
        values[15] = (-iSp + iss)/tiSp
        ip = AiF*iFp + AiS*iSp
        Gto = (4.0*Gto_b if celltype == 1.0 else (4.0*Gto_b if celltype == 2.0 else\
            Gto_b))
        fItop = 1.0/(1.0 + KmCaMK/CaMKa)
        Ito = (-EK + v)*((1.0 - fItop)*a*i + ap*fItop*ip)*Gto

        # Expressions for the ICaL component
        dss = 1.0/(1.0 + 0.39398514226669484*math.exp(-0.23640661938534277*v))
        td = 0.6 + 1.0/(3.5254214873653824*math.exp(0.09*v) +\
            0.7408182206817179*math.exp(-0.05*v))
        values[16] = (-d + dss)/td
        fss = 1.0/(1.0 + 199.86038496778565*math.exp(0.27056277056277056*v))
        tff = 7.0 + 1.0/(0.03325075244518792*math.exp(0.1*v) +\
            0.0006090087745647571*math.exp(-0.1*v))
        tfs = 1000.0 + 1.0/(1.0027667890106652e-05*math.exp(-0.25*v) +\
            8.053415618124885e-05*math.exp(0.16666666666666666*v))
        Aff = 0.6
        Afs = 1.0 - Aff
        values[17] = (-ff + fss)/tff
        values[18] = (-fs + fss)/tfs
        f = Aff*ff + Afs*fs
        fcass = fss
        tfcaf = 7.0 + 1.0/(0.0708317980974062*math.exp(-0.14285714285714285*v) +\
            0.02258872488031037*math.exp(0.14285714285714285*v))
        tfcas = 100.0 + 1.0/(0.00012*math.exp(0.14285714285714285*v) +\
            0.00012*math.exp(-0.3333333333333333*v))
        Afcaf = 0.3 + 0.6/(1.0 + 0.36787944117144233*math.exp(0.1*v))
        Afcas = 1.0 - Afcaf
        values[19] = (-fcaf + fcass)/tfcaf
        values[20] = (-fcas + fcass)/tfcas
        fca = Afcaf*fcaf + Afcas*fcas
        tjca = 75.0
        values[21] = (-jca + fcass)/tjca
        tffp = 2.5*tff
        values[22] = (-ffp + fss)/tffp
        fp = Aff*ffp + Afs*fs
        tfcafp = 2.5*tfcaf
        values[23] = (-fcafp + fcass)/tfcafp
        fcap = Afcaf*fcafp + Afcas*fcas
        km2n = 1.0*jca
        anca = 1.0/(math.pow(1.0 + Kmn/cass, 4.0) + k2n/km2n)
        values[24] = k2n*anca - km2n*nca
        v0 = 0
        B_1 = 2.0*frt
        A_1 = 4.0*(-0.341*cao + cass*math.exp(2.0*vfrt))*ffrt/B_1
        U_1 = (-v0 + v)*B_1
        PhiCaL = ((1.0 - 0.5*U_1)*A_1 if U_1 <= 1e-07 and -1e-07 <= U_1 else\
            A_1*U_1/(-1.0 + math.exp(U_1)))
        B_2 = frt
        A_2 = 0.75*(-nao + math.exp(vfrt)*nass)*ffrt/B_2
        U_2 = (-v0 + v)*B_2
        PhiCaNa = ((1.0 - 0.5*U_2)*A_2 if U_2 <= 1e-07 and -1e-07 <= U_2 else\
            A_2*U_2/(-1.0 + math.exp(U_2)))
        B_3 = frt
        A_3 = 0.75*(-ko + math.exp(vfrt)*kss)*ffrt/B_3
        U_3 = (-v0 + v)*B_3
        PhiCaK = ((1.0 - 0.5*U_3)*A_3 if U_3 <= 1e-07 and -1e-07 <= U_3 else\
            A_3*U_3/(-1.0 + math.exp(U_3)))
        PCa = (1.2*PCa_b if celltype == 1.0 else (2.5*PCa_b if celltype == 2.0 else\
            PCa_b))
        PCap = 1.1*PCa
        PCaNa = 0.00125*PCa
        PCaK = 0.0003574*PCa
        PCaNap = 0.00125*PCap
        PCaKp = 0.0003574*PCap
        fICaLp = 1.0/(1.0 + KmCaMK/CaMKa)
        ICaL = (1.0 - fICaLp)*((1.0 - nca)*f + fca*jca*nca)*PCa*PhiCaL*d + ((1.0 -\
            nca)*fp + fcap*jca*nca)*PCap*PhiCaL*d*fICaLp
        ICaNa = (1.0 - fICaLp)*((1.0 - nca)*f + fca*jca*nca)*PCaNa*PhiCaNa*d +\
            ((1.0 - nca)*fp + fcap*jca*nca)*PCaNap*PhiCaNa*d*fICaLp
        ICaK = (1.0 - fICaLp)*((1.0 - nca)*f + fca*jca*nca)*PCaK*PhiCaK*d + ((1.0 -\
            nca)*fp + fcap*jca*nca)*PCaKp*PhiCaK*d*fICaLp

        # Expressions for the IKr component
        values[25] = A21*IC2*math.exp(B21*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q21)) + A51*C1*math.exp(B51*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q51)) - A11*IC1*math.exp(B11*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q11)) - A61*IC1*math.exp(B61*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q61))
        values[26] = A11*IC1*math.exp(B11*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q11)) + A4*IO*math.exp(B4*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q4)) + A52*C2*math.exp(B52*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q52)) - A21*IC2*math.exp(B21*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q21)) - A3*IC2*math.exp(B3*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q3)) - A62*IC2*math.exp(B62*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q62))
        values[27] = A2*C2*math.exp(B2*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q2)) + A61*IC1*math.exp(B61*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q61)) - A1*C1*math.exp(B1*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q1)) - A51*C1*math.exp(B51*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q51))
        values[28] = A1*C1*math.exp(B1*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q1)) + A41*O*math.exp(B41*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q41)) + A62*IC2*math.exp(B62*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q62)) - A2*C2*math.exp(B2*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q2)) - A31*C2*math.exp(B31*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q31)) - A52*C2*math.exp(B52*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q52))
        if (D == 0):
            D = .0001
        values[29] = Ku*Obound + A31*C2*math.exp(B31*v)*math.exp(0.1*(-20.0 +Temp)*math.log(q31)) + A63*IO*math.exp(B63*v)*math.exp(0.1*(-20.0 + Temp)*math.log(q63)) - A41*O*math.exp(B41*v)*math.exp(0.1*(-20.0 + Temp)*math.log(q41)) - A53*O*math.exp(B53*v)*math.exp(0.1*(-20.0 + Temp)*math.log(q53)) - Kmax*Ku*O*math.exp(n*math.log(D))/(halfmax + math.exp(n*math.log(D)))
        try: 
            exp_b63_v = (math.exp(-B63*v))
        except:
            exp_b63_v= 0
        values[30] = A3*IC2*math.exp(B3*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q3)) + A53*O*math.exp(B53*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q53)) - A4*IO*math.exp(B4*v)*math.exp(0.1*(-20.0 +\
            Temp)*math.log(q4)) - A63*IO*math.exp(B63*v)*math.exp(0.1*(-20.0 + Temp)*math.log(q63)) - Kmax*Ku*IO*math.exp(n*math.log(D))/(halfmax + math.exp(n*math.log(D))) + \
            A53*Ku*IObound*math.exp(B53*v)*exp_b63_v*math.exp(0.1*(-20.0 + Temp)*math.log(q53))*math.exp(-0.1*(-20.0 + Temp)*math.log(q63))/A63
        values[31] = -Kt*IObound + Kt*Cbound/(1.0 +\
            math.exp(0.14729709824716453*Vhalf - 0.14729709824716453*v)) +\
            Kmax*Ku*IO*math.exp(n*math.log(D))/(halfmax +\
            math.exp(n*math.log(D))) -\
            A53*Ku*IObound*math.exp(B53*v)*exp_b63_v*math.exp(0.1*(-20.0 + Temp)*math.log(q53))*math.exp(-0.1*(-20.0 + Temp)*math.log(q63))/A63
        values[32] = -Kt*Obound - Ku*Obound + Kt*Cbound/(1.0 +\
            math.exp(0.14729709824716453*Vhalf - 0.14729709824716453*v)) +\
            Kmax*Ku*O*math.exp(n*math.log(D))/(halfmax + math.exp(n*math.log(D)))
        values[33] = Kt*IObound + Kt*Obound - 2*Kt*Cbound/(1.0 +\
            math.exp(0.14729709824716453*Vhalf - 0.14729709824716453*v))
        values[34] = 0
        GKr = (1.3*GKr_b if celltype == 1.0 else (0.8*GKr_b if celltype == 2.0 else\
            GKr_b))
        IKr = 0.4303314829119352*math.sqrt(ko)*(-EK + v)*GKr*O

        # Expressions for the IKs component
        xs1ss = 1.0/(1.0 + 0.27288596035656526*math.exp(-0.11195700850873264*v))
        txs1 = txs1_max +\
            1.0/(0.003504067763074858*math.exp(0.056179775280898875*v) +\
            0.0005184809083581659*math.exp(-0.004347826086956522*v))
        values[35] = (-xs1 + xs1ss)/txs1
        xs2ss = xs1ss
        txs2 = 1.0/(0.0022561357010639103*math.exp(-0.03225806451612903*v) +\
            0.0008208499862389881*math.exp(0.05*v))
        values[36] = (-xs2 + xs2ss)/txs2
        KsCa = 1.0 + 0.6/(1.0 + 6.481821026062645e-07*math.pow(1.0/cai, 1.4))
        GKs = (1.4*GKs_b if celltype == 1.0 else GKs_b)
        IKs = (-EKs + v)*GKs*KsCa*xs1*xs2

        # Expressions for the IK1 component
        xk1ss = 1.0/(1.0 + math.exp((-144.59 - v - 2.5538*ko)/(3.8115 +\
            1.5692*ko)))
        txk1 = 122.2/(0.0019352007631390235*math.exp(-0.049115913555992145*v) +\
            30.43364757524903*math.exp(0.014423770373575654*v))
        values[37] = (-xk1 + xk1ss)/txk1
        rk1 = 1.0/(1.0 + 69220.6322106767*math.exp(0.10534077741493732*v -\
            0.27388602127883704*ko))
        GK1 = (1.2*GK1_b if celltype == 1.0 else (1.3*GK1_b if celltype == 2.0 else\
            GK1_b))
        IK1 = math.sqrt(ko)*(-EK + v)*GK1*rk1*xk1

        # Expressions for the Ca i component
        hca = math.exp(F*qca*v/(R*T))
        hna = math.exp(F*qna*v/(R*T))
        h1_i = 1.0 + (1.0 + hna)*nai/kna3
        h2_i = hna*nai/(kna3*h1_i)
        h3_i = 1.0/h1_i
        h4_i = 1.0 + (1.0 + nai/kna2)*nai/kna1
        h5_i = (nai*nai)/(kna1*kna2*h4_i)
        h6_i = 1.0/h4_i
        h7_i = 1.0 + nao*(1.0 + 1.0/hna)/kna3
        h8_i = nao/(kna3*h7_i*hna)
        h9_i = 1.0/h7_i
        h10_i = 1.0 + kasymm + nao*(1.0 + nao/kna2)/kna1
        h11_i = (nao*nao)/(kna1*kna2*h10_i)
        h12_i = 1.0/h10_i
        k1_i = cao*kcaon*h12_i
        k2_i = kcaoff
        k3p_i = wca*h9_i
        k3pp_i = wnaca*h8_i
        k3_i = k3p_i + k3pp_i
        k4p_i = wca*h3_i/hca
        k4pp_i = wnaca*h2_i
        k4_i = k4p_i + k4pp_i
        k5_i = kcaoff
        k6_i = kcaon*cai*h6_i
        k7_i = wna*h2_i*h5_i
        k8_i = wna*h11_i*h8_i
        x1_i = (k2_i + k3_i)*k5_i*k7_i + (k6_i + k7_i)*k2_i*k4_i
        x2_i = (k1_i + k8_i)*k4_i*k6_i + (k4_i + k5_i)*k1_i*k7_i
        x3_i = (k2_i + k3_i)*k6_i*k8_i + (k6_i + k7_i)*k1_i*k3_i
        x4_i = (k1_i + k8_i)*k3_i*k5_i + (k4_i + k5_i)*k2_i*k8_i
        E1_i = x1_i/(x1_i + x2_i + x3_i + x4_i)
        E2_i = x2_i/(x1_i + x2_i + x3_i + x4_i)
        E3_i = x3_i/(x1_i + x2_i + x3_i + x4_i)
        E4_i = x4_i/(x1_i + x2_i + x3_i + x4_i)
        allo_i = 1.0/(1.0 + math.pow(KmCaAct/cai, 2.0))
        JncxNa_i = E3_i*k4pp_i - E2_i*k3pp_i + 3.0*E4_i*k7_i - 3.0*E1_i*k8_i
        JncxCa_i = E2_i*k2_i - E1_i*k1_i
        Gncx = (1.1*Gncx_b if celltype == 1.0 else (1.4*Gncx_b if celltype == 2.0 else\
            Gncx_b))
        INaCa_i = 0.8*(zca*JncxCa_i + zna*JncxNa_i)*Gncx*allo_i
        h1_ss = 1.0 + (1.0 + hna)*nass/kna3
        h2_ss = hna*nass/(kna3*h1_ss)
        h3_ss = 1.0/h1_ss
        h4_ss = 1.0 + (1.0 + nass/kna2)*nass/kna1
        h5_ss = (nass*nass)/(kna1*kna2*h4_ss)
        h6_ss = 1.0/h4_ss
        h7_ss = 1.0 + nao*(1.0 + 1.0/hna)/kna3
        h8_ss = nao/(kna3*h7_ss*hna)
        h9_ss = 1.0/h7_ss
        h10_ss = 1.0 + kasymm + nao*(1.0 + nao/kna2)/kna1
        h11_ss = (nao*nao)/(kna1*kna2*h10_ss)
        h12_ss = 1.0/h10_ss
        k1_ss = cao*kcaon*h12_ss
        k2_ss = kcaoff
        k3p_ss = wca*h9_ss
        k3pp_ss = wnaca*h8_ss
        k3_ss = k3p_ss + k3pp_ss
        k4p_ss = wca*h3_ss/hca
        k4pp_ss = wnaca*h2_ss
        k4_ss = k4p_ss + k4pp_ss
        k5_ss = kcaoff
        k6_ss = kcaon*cass*h6_ss
        k7_ss = wna*h2_ss*h5_ss
        k8_ss = wna*h11_ss*h8_ss
        x1_ss = (k2_ss + k3_ss)*k5_ss*k7_ss + (k6_ss + k7_ss)*k2_ss*k4_ss
        x2_ss = (k1_ss + k8_ss)*k4_ss*k6_ss + (k4_ss + k5_ss)*k1_ss*k7_ss
        x3_ss = (k2_ss + k3_ss)*k6_ss*k8_ss + (k6_ss + k7_ss)*k1_ss*k3_ss
        x4_ss = (k1_ss + k8_ss)*k3_ss*k5_ss + (k4_ss + k5_ss)*k2_ss*k8_ss
        E1_ss = x1_ss/(x1_ss + x2_ss + x3_ss + x4_ss)
        E2_ss = x2_ss/(x1_ss + x2_ss + x3_ss + x4_ss)
        E3_ss = x3_ss/(x1_ss + x2_ss + x3_ss + x4_ss)
        E4_ss = x4_ss/(x1_ss + x2_ss + x3_ss + x4_ss)
        allo_ss = 1.0/(1.0 + math.pow(KmCaAct/cass, 2.0))
        JncxNa_ss = E3_ss*k4pp_ss - E2_ss*k3pp_ss + 3.0*E4_ss*k7_ss -\
            3.0*E1_ss*k8_ss
        JncxCa_ss = E2_ss*k2_ss - E1_ss*k1_ss
        INaCa_ss = 0.2*(zca*JncxCa_ss + zna*JncxNa_ss)*Gncx*allo_ss

        # Expressions for the K component
        Knai = Knai0*math.exp(0.3333333333333333*F*delta*v/(R*T))
        Knao = Knao0*math.exp(0.3333333333333333*F*(1.0 - delta)*v/(R*T))
        P = eP/(1.0 + H/Khp + nai/Knap + ki/Kxkur)
        a1 = k1p*math.pow(nai/Knai, 3.0)/(-1.0 + math.pow(1.0 + ki/Kki, 2.0) +\
            math.pow(1.0 + nai/Knai, 3.0))
        b1 = MgADP*k1m
        a2 = k2p
        b2 = k2m*math.pow(nao/Knao, 3.0)/(-1.0 + math.pow(1.0 + ko/Kko, 2.0) +\
            math.pow(1.0 + nao/Knao, 3.0))
        a3 = k3p*math.pow(ko/Kko, 2.0)/(-1.0 + math.pow(1.0 + ko/Kko, 2.0) +\
            math.pow(1.0 + nao/Knao, 3.0))
        b3 = H*k3m*P/(1.0 + MgATP/Kmgatp)
        a4 = MgATP*k4p/(Kmgatp*(1.0 + MgATP/Kmgatp))
        b4 = k4m*math.pow(ki/Kki, 2.0)/(-1.0 + math.pow(1.0 + ki/Kki, 2.0) +\
            math.pow(1.0 + nai/Knai, 3.0))
        x1 = a1*a2*a4 + a1*a2*b3 + a2*b3*b4 + b2*b3*b4
        x2 = a1*a2*a3 + a2*a3*b4 + a3*b1*b4 + b1*b2*b4
        x3 = a2*a3*a4 + a3*a4*b1 + a4*b1*b2 + b1*b2*b3
        x4 = a1*a3*a4 + a1*a4*b2 + a1*b2*b3 + b2*b3*b4
        E1 = x1/(x1 + x2 + x3 + x4)
        E2 = x2/(x1 + x2 + x3 + x4)
        E3 = x3/(x1 + x2 + x3 + x4)
        E4 = x4/(x1 + x2 + x3 + x4)
        JnakNa = 3.0*E1*a3 - 3.0*E2*b3
        JnakK = 2.0*E4*b1 - 2.0*E3*a1
        Pnak = (0.9*Pnak_b if celltype == 1.0 else (0.7*Pnak_b if celltype == 2.0 else\
            Pnak_b))
        INaK = (zk*JnakK + zna*JnakNa)*Pnak

        # Expressions for the IKb component
        xkb = 1.0/(1.0 + 2.202363450949239*math.exp(-0.05452562704471101*v))
        GKb = (0.6*GKb_b if celltype == 1.0 else GKb_b)
        IKb = (-EK + v)*GKb*xkb

        # Expressions for the b component
        B = frt
        v0 = 0
        A = PNab*(-nao + math.exp(vfrt)*nai)*ffrt/B
        U = (-v0 + v)*B
        INab = ((1.0 - 0.5*U)*A if U <= 1e-07 and -1e-07 <= U else A*U/(-1.0 +\
            math.exp(U)))

        # Expressions for the ICab component
        B = 2.0*frt
        v0 = 0
        A = 4.0*PCab*(-0.341*cao + cai*math.exp(2.0*vfrt))*ffrt/B
        U = (-v0 + v)*B
        ICab = ((1.0 - 0.5*U)*A if U <= 1e-07 and -1e-07 <= U else A*U/(-1.0 +\
            math.exp(U)))

        # Expressions for the IpCa component
        IpCa = GpCa*cai/(KmCap + cai)

        # Expressions for the Diff component
        JdiffNa = 0.5*nass - 0.5*nai
        JdiffK = 0.5*kss - 0.5*ki
        Jdiff = 5.0*cass - 5.0*cai

        # Expressions for the Ryr component
        a_rel = 0.5*bt
        Jrel_inf_temp = -ICaL*a_rel/(1.0 + 25.62890625*math.pow(1.0/cajsr, 8.0))
        Jrel_inf = (1.7*Jrel_inf_temp if celltype == 2.0 else Jrel_inf_temp)
        tau_rel_temp = bt/(1.0 + 0.0123/cajsr)
        tau_rel = (0.001 if tau_rel_temp < 0.001 else tau_rel_temp)
        values[38] = (-Jrelnp + Jrel_inf)/tau_rel
        btp = 1.25*bt
        a_relp = 0.5*btp
        Jrel_temp = -ICaL*a_relp/(1.0 + 25.62890625*math.pow(1.0/cajsr, 8.0))
        Jrel_infp = (1.7*Jrel_temp if celltype == 2.0 else Jrel_temp)
        tau_relp_temp = btp/(1.0 + 0.0123/cajsr)
        tau_relp = (0.001 if tau_relp_temp < 0.001 else tau_relp_temp)
        values[39] = (-Jrelp + Jrel_infp)/tau_relp
        fJrelp = 1.0/(1.0 + KmCaMK/CaMKa)
        Jrel = Jrel_scaling_factor*((1.0 - fJrelp)*Jrelnp + Jrelp*fJrelp)

        # Expressions for the SERCA component
        upScale = (1.3 if celltype == 1.0 else 1.0)
        Jupnp = 0.004375*cai*upScale/(0.00092 + cai)
        Jupp = 0.01203125*cai*upScale/(0.00075 + cai)
        fJupp = 1.0/(1.0 + KmCaMK/CaMKa)
        Jleak = 0.0002625*cansr
        Jup = Jup_b*(-Jleak + (1.0 - fJupp)*Jupnp + Jupp*fJupp)

        # Expressions for the Trans flux component
        Jtr = 0.01*cansr - 0.01*cajsr

        Istim = -self.i_stimulation

        values[40] = -ICaK - ICaL - ICaNa - ICab - IK1 - IKb - IKr - IKs - INa -\
            INaCa_i - INaCa_ss - INaK - INaL - INab - IpCa - Istim - Ito

        if self.current_response_info:
            current_timestep = [
                trace.Current(name='I_K1', value=IK1),
                trace.Current(name='I_To', value=Ito),
                trace.Current(name='I_Kr', value=IKr),
                trace.Current(name='I_Ks', value=IKs),
                trace.Current(name='I_CaL', value=ICaL),
                trace.Current(name='I_NaK', value=INaK),
                trace.Current(name='I_Na', value=INa),
                trace.Current(name='I_NaCa', value=INaCa_i),
                trace.Current(name='I_pCa', value=IpCa),
                trace.Current(name='I_bNa', value=INab),
                trace.Current(name='I_bCa', value=ICab),
                trace.Current(name='I_CaK', value=ICaK),
                trace.Current(name='I_CaNa', value=ICaNa),
                trace.Current(name='I_Cab', value=ICab),
                trace.Current(name='I_Kb', value=IKb),
                trace.Current(name='I_NaCa_ss', value=INaCa_ss),
                trace.Current(name='I_NaL', value=INaL)
            ]
            self.current_response_info.currents.append(current_timestep)

        # Expressions for the Intracellular ions component
        cmdnmax = (1.3*cmdnmax_b if celltype == 1.0 else cmdnmax_b)
        values[41] = JdiffNa*vss/vmyo + cm*(-INa - INaL - INab - 3.0*INaCa_i -\
            3.0*INaK)*Acap/(F*vmyo)
        values[42] = -JdiffNa + cm*(-ICaNa - 3.0*INaCa_ss)*Acap/(F*vss)
        values[43] = JdiffK*vss/vmyo + cm*(-IK1 - IKb - IKr - IKs - Istim - Ito +\
            2.0*INaK)*Acap/(F*vmyo)
        values[44] = -JdiffK - cm*Acap*ICaK/(F*vss)
        Bcai = 1.0/(1.0 + kmcmdn*math.pow(kmcmdn + cai, -2.0)*cmdnmax +\
            kmtrpn*trpnmax*math.pow(kmtrpn + cai, -2.0))
        values[45] = (Jdiff*vss/vmyo - Jup*vnsr/vmyo + 0.5*cm*(-ICab - IpCa +\
            2.0*INaCa_i)*Acap/(F*vmyo))*Bcai
        Bcass = 1.0/(1.0 + BSLmax*KmBSL*math.pow(KmBSL + cass, -2.0) +\
            BSRmax*KmBSR*math.pow(KmBSR + cass, -2.0))
        values[46] = (-Jdiff + Jrel*vjsr/vss + 0.5*cm*(-ICaL +\
            2.0*INaCa_ss)*Acap/(F*vss))*Bcass
        values[47] = -Jtr*vjsr/vnsr + Jup
        Bcajsr = 1.0/(1.0 + csqnmax*kmcsqn*math.pow(kmcsqn + cajsr, -2.0))
        values[48] = (-Jrel + Jtr)*Bcajsr

        # Return results
        return values

