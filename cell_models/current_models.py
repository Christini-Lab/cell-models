import numpy as np
from math import log, sqrt, exp

from cell_models.model_initial import kernik_model_inputs


class KernikCurrents():
    """An implementation of all Kernik currents by Kernik et al.

    Attributes:
        default_parameters: A dict containing tunable parameters along with
            their default values as specified in Kernik et al.
        updated_parameters: A dict containing all parameters that are being
            tuned.
    """

    # Load model parameters
    model_parameter_inputs = kernik_model_inputs()

    # Current parameter values:
    x_K1 = model_parameter_inputs[16:22]
    x_KR = model_parameter_inputs[22:33]
    x_IKS = model_parameter_inputs[33:39]
    xTO = model_parameter_inputs[39:50]
    x_cal = model_parameter_inputs[50:61]
    x_cat = model_parameter_inputs[61]
    x_NA = model_parameter_inputs[62:76]
    x_F = model_parameter_inputs[76:82]

    Cm = 60
    V_tot = 3960
    Vc_tenT = 16404
    VSR_tenT = 1094
    V_tot_tenT = Vc_tenT + VSR_tenT
    Vc = V_tot * (Vc_tenT / V_tot_tenT)
    V_SR = V_tot * (VSR_tenT / V_tot_tenT)

    def __init__(self, t_kelvin=310.0, f_coulomb_per_mmole=96.4853415,
                 r_joule_per_mole_kelvin=8.314472):
        self.t_kelvin = t_kelvin  
        self.r_joule_per_mole_kelvin = r_joule_per_mole_kelvin  
        self.f_coulomb_per_mmole = f_coulomb_per_mmole 

        self.Ko = 5.4  # millimolar (in model_parameters)
        self.Cao = 1.8  # millimolar (in model_parameters
        self.Nao = 140.0  # millimolar (in model_parameters)

    def i_K1(self, v_m, E_K, g_K1):
        xK11 = self.x_K1[1]
        xK12 = self.x_K1[2]
        xK13 = self.x_K1[3]
        xK14 = self.x_K1[4]
        xK15 = self.x_K1[5]

        alpha_xK1 = xK11*exp((v_m+xK13)/xK12)
        beta_xK1 = exp((v_m+xK15)/xK14)
        XK1_inf = alpha_xK1/(alpha_xK1+beta_xK1)

        # Current:
        g_K1 = self.x_K1[0] * g_K1 
        return g_K1*XK1_inf*(v_m-E_K)*sqrt(self.Ko/5.4)

    def i_Kr(self, v_m, E_K, Xr1, Xr2, g_Kr):
        # define parameters from x_KR
        Xr1_1 = self.x_KR[1]
        Xr1_2 = self.x_KR[2]
        Xr1_5 = self.x_KR[3]
        Xr1_6 = self.x_KR[4]
        Xr2_1 = self.x_KR[5]
        Xr2_2 = self.x_KR[6]
        Xr2_5 = self.x_KR[7]
        Xr2_6 = self.x_KR[8]

        # parameter-dependent values:
        Xr1_3 = Xr1_5*Xr1_1
        Xr2_3 = Xr2_5*Xr2_1
        Xr1_4 = 1/((1/Xr1_2)+(1/Xr1_6))
        Xr2_4 = 1/((1/Xr2_2)+(1/Xr2_6))

        # 10: Xr1 (dimensionless) (activation in i_Kr_Xr1)
        alpha_Xr1 = Xr1_1*exp((v_m)/Xr1_2)
        beta_Xr1 = Xr1_3*exp((v_m)/Xr1_4)
        Xr1_inf = alpha_Xr1/(alpha_Xr1 + beta_Xr1)
        tau_Xr1 = ((1./(alpha_Xr1 + beta_Xr1))+self.x_KR[9])
        d_Xr1 = (Xr1_inf-Xr1)/tau_Xr1

        # 11: Xr2 (dimensionless) (inactivation in i_Kr_Xr2)
        alpha_Xr2 = Xr2_1*exp((v_m)/Xr2_2)
        beta_Xr2 = Xr2_3*exp((v_m)/Xr2_4)
        Xr2_inf = alpha_Xr2/(alpha_Xr2+beta_Xr2)
        tau_Xr2 = ((1./(alpha_Xr2+beta_Xr2))+self.x_KR[10])
        d_Xr2 = (Xr2_inf-Xr2)/tau_Xr2

        # Current:
        g_Kr = self.x_KR[0]*g_Kr  # nS_per_pF (in i_Kr)
        i_Kr = g_Kr*(v_m-E_K)*Xr1*Xr2*sqrt(self.Ko/5.4)
        return [d_Xr1, d_Xr2, i_Kr]

    def i_Ks(self, v_m, E_K, Xs, g_Ks):
        ks1 = self.x_IKS[1]
        ks2 = self.x_IKS[2]
        ks5 = self.x_IKS[3]
        ks6 = self.x_IKS[4]
        tauks_const = self.x_IKS[5]

        # parameter-dependent values:
        ks3 = ks5*ks1
        ks4 = 1/((1/ks2)+(1/ks6))

        # 12: Xs (dimensionless) (activation in i_Ks)
        alpha_Xs = ks1*exp((v_m)/ks2)
        beta_Xs = ks3*exp((v_m)/ks4)
        Xs_inf = alpha_Xs/(alpha_Xs+beta_Xs)
        tau_Xs = (1./(alpha_Xs+beta_Xs)) + tauks_const
        d_Xs = (Xs_inf-Xs)/tau_Xs

        # Current:
        g_Ks = self.x_IKS[0]*g_Ks  # nS_per_pF (in i_Ks)
        i_Ks = g_Ks*(v_m-E_K)*(Xs**2)

        return [d_Xs, i_Ks]

    def i_to(self, v_m, E_K, s, r, g_to):
        # Transient outward current (Ito): define parameters from xTO
        r1 = self.xTO[1]
        r2 = self.xTO[2]
        r5 = self.xTO[3]
        r6 = self.xTO[4]
        s1 = self.xTO[5]
        s2 = self.xTO[6]
        s5 = self.xTO[7]
        s6 = self.xTO[8]
        tau_r_const = self.xTO[9]
        tau_s_const = self.xTO[10]

        # parameter-dependent values:
        r3 = r5*r1
        r4 = 1/((1/r2)+(1/r6))
        s3 = s5*s1
        s4 = 1/((1/s2)+(1/s6))

        # 17: s (dimensionless) (inactivation in i_to)
        alpha_s = s1*exp((v_m)/s2)
        beta_s = s3*exp((v_m)/s4)
        s_inf = alpha_s/(alpha_s+beta_s)
        tau_s = ((1./(alpha_s+beta_s))+tau_s_const)
        d_s = (s_inf-s)/tau_s

        # 18: r (dimensionless) (activation in i_to)
        alpha_r = r1*exp((v_m)/r2)
        beta_r = r3*exp((v_m)/r4)
        r_inf = alpha_r/(alpha_r + beta_r)
        tau_r = (1./(alpha_r + beta_r))+tau_r_const
        d_r = (r_inf-r)/tau_r

        # Current:
        g_to = self.xTO[0]*g_to  # nS_per_pF (in i_to)
        i_to = g_to*(v_m-E_K)*s*r
        return [d_s, d_r, i_to]

    def i_CaL(self, v_m, d, f, fCa, Cai, Nai, Ki, p_CaL):
        d1 = self.x_cal[1]
        d2 = self.x_cal[2]
        d5 = self.x_cal[3]
        d6 = self.x_cal[4]
        f1 = self.x_cal[5]
        f2 = self.x_cal[6]
        f5 = self.x_cal[7]
        f6 = self.x_cal[8]
        taud_const = self.x_cal[9]
        tauf_const = self.x_cal[10]

        # parameter-dependent values:
        d3 = d5*d1
        d4 = 1/((1/d2)+(1/d6))
        f3 = f5*f1
        f4 = 1/((1/f2)+(1/f6))

        # 7: d (dimensionless) (activation in i_CaL)
        alpha_d = d1*exp(((v_m))/d2)
        beta_d = d3*exp(((v_m))/d4)
        d_inf = alpha_d/(alpha_d + beta_d)
        tau_d = ((1/(alpha_d + beta_d))+taud_const)
        d_d = (d_inf-d)/tau_d

        # 8: f (dimensionless) (inactivation  i_CaL)
        alpha_f = f1*exp(((v_m))/f2)
        beta_f = f3*exp(((v_m))/f4)
        f_inf = alpha_f/(alpha_f+beta_f)
        tau_f = ((1./(alpha_f+beta_f)) + tauf_const)
        d_f = (f_inf-f)/tau_f

        # 9: fCa (dimensionless) (calcium-dependent inactivation in i_CaL)
        # from Ten tusscher 2004
        scale_Ical_Fca_Cadep = 1.2
        alpha_fCa = 1.0/(1.0+((scale_Ical_Fca_Cadep*Cai)/.000325) ** 8.0)

        try:
            beta_fCa = 0.1/(1.0+exp((scale_Ical_Fca_Cadep*Cai-.0005)/0.0001))
        except OverflowError:
            beta_fCa_exp = (scale_Ical_Fca_Cadep*Cai-.0005)/0.0001

            if beta_fCa_exp > 50:
                beta_fCa = 0
            else:
                beta_fCa = 0.1

        gamma_fCa = .2/(1.0+exp((scale_Ical_Fca_Cadep*Cai-0.00075)/0.0008))

        fCa_inf = ((alpha_fCa+beta_fCa+gamma_fCa+.23)/(1.46))
        tau_fCa = 2  # ms
        if ((fCa_inf > fCa) and (v_m > -60)):
            k_fca = 0
        else:
            k_fca = 1

        d_fCa = k_fca*(fCa_inf-fCa)/tau_fCa

        # Current
        p_CaL = self.x_cal[0]*p_CaL  # nS_per_pF (in i_CaL)
        p_CaL_shannonCa = 5.4e-4
        p_CaL_shannonNa = 1.5e-8
        p_CaL_shannonK = 2.7e-7
        p_CaL_shannonTot = p_CaL_shannonCa + p_CaL_shannonNa + p_CaL_shannonK
        p_CaL_shannonCap = p_CaL_shannonCa/p_CaL_shannonTot
        p_CaL_shannonNap = p_CaL_shannonNa/p_CaL_shannonTot
        p_CaL_shannonKp = p_CaL_shannonK/p_CaL_shannonTot

        p_CaL_Ca = p_CaL_shannonCap*p_CaL
        p_CaL_Na = p_CaL_shannonNap*p_CaL
        p_CaL_K = p_CaL_shannonKp*p_CaL

        ibarca = p_CaL_Ca*4.0*v_m*self.f_coulomb_per_mmole ** 2.0/(self.r_joule_per_mole_kelvin*self.t_kelvin) * (.341*Cai*exp(
            2.0*v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin))-0.341*self.Cao)/(exp(2.0*v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin))-1.0)
        i_CaL_Ca = ibarca * d*f*fCa

        ibarna = p_CaL_Na * \
            v_m*self.f_coulomb_per_mmole ** 2.0/(self.r_joule_per_mole_kelvin*self.t_kelvin) * (.75*Nai*exp(v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin)) -
                                  0.75*self.Nao)/(exp(v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin))-1.0)
        i_CaL_Na = ibarna * d*f*fCa

        ibark = p_CaL_K*v_m*self.f_coulomb_per_mmole ** 2.0/(self.r_joule_per_mole_kelvin*self.t_kelvin) * (.75*Ki *
                                              exp(v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin))-0.75*self.Ko)/(exp(
                                                  v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin))-1.0)
        i_CaL_K = ibark * d*f*fCa

        i_CaL = i_CaL_Ca+i_CaL_Na+i_CaL_K

        return [d_d, d_f, d_fCa, i_CaL, i_CaL_Ca, i_CaL_Na, i_CaL_K]

    def i_CaT(self, v_m, E_Ca, dCaT, fCaT, g_CaT):
        # 19: dCaT (activation in i_CaT)
        dcat_inf = 1./(1+exp(-((v_m) + 26.3)/6))
        tau_dcat = 1./(1.068*exp(((v_m)+26.3)/30) + 1.068*exp(-((v_m)+26.3)/30))
        d_dCaT = (dcat_inf-dCaT)/tau_dcat

        # 20: fCaT (inactivation in i_CaT)
        fcat_inf = 1./(1+exp(((v_m) + 61.7)/5.6))
        tau_fcat = 1./(.0153*exp(-((v_m)+61.7)/83.3) + 0.015*exp(
            ((v_m)+61.7)/15.38))
        d_fCaT = (fcat_inf-fCaT)/tau_fcat

        g_CaT = self.x_cat*g_CaT # nS_per_pF (in i_CaT)
        i_CaT = g_CaT*(v_m-E_Ca)*dCaT*fCaT

        return [d_dCaT, d_fCaT, i_CaT]

    def i_Na(self, v_m, E_Na, h, j, m, g_Na):
        # Sodium Current (INa):
        # define parameters from x_Na
        m1 = self.x_NA[1]
        m2 = self.x_NA[2]
        m5 = self.x_NA[3]
        m6 = self.x_NA[4]
        h1 = self.x_NA[5]
        h2 = self.x_NA[6]
        h5 = self.x_NA[7]
        h6 = self.x_NA[8]
        j1 = self.x_NA[9]
        j2 = self.x_NA[10]
        tau_m_const = self.x_NA[11]
        tau_h_const = self.x_NA[12]
        tau_j_const = self.x_NA[13]

        # parameter-dependent values:
        m3 = m5*m1
        m4 = 1/((1/m2)+(1/m6))
        h3 = h5*h1
        h4 = 1/((1/h2)+(1/h6))
        j5 = h5
        j6 = h6
        j3 = j5*j1
        j4 = 1/((1/j2)+(1/j6))

        # 13: h (dimensionless) (inactivation in i_Na)
        alpha_h = h1*exp((v_m)/h2)
        beta_h = h3*exp((v_m)/h4)
        h_inf = (alpha_h/(alpha_h+beta_h))
        tau_h = ((1./(alpha_h+beta_h))+tau_h_const)
        d_h = (h_inf-h)/tau_h

        # 14: j (dimensionless) (slow inactivation in i_Na)
        alpha_j = j1*exp((v_m)/j2)
        beta_j = j3*exp((v_m)/j4)
        j_inf = (alpha_j/(alpha_j+beta_j))
        tau_j = ((1./(alpha_j+beta_j))+tau_j_const)
        d_j = (j_inf-j)/tau_j

        # 15: m (dimensionless) (activation in i_Na)
        alpha_m = m1*exp((v_m)/m2)
        beta_m = m3*exp((v_m)/m4)
        m_inf = alpha_m/(alpha_m+beta_m)
        tau_m = ((1./(alpha_m+beta_m))+tau_m_const)
        d_m = (m_inf-m)/tau_m

        # Current:
        g_Na = self.x_NA[0]*g_Na
        # nS_per_pF (in i_Na)
        i_Na = g_Na*m ** 3.0*h*j*(v_m-E_Na)

        return [d_h, d_j, d_m, i_Na]

    def i_f(self, v_m, E_K, E_Na, Xf, g_f):
        # Funny/HCN current (If):
        # define parameters from x_F
        xF1 = self.x_F[1]
        xF2 = self.x_F[2]
        xF5 = self.x_F[3]
        xF6 = self.x_F[4]
        xF_const = self.x_F[5]

        # parameter-dependent values:
        xF3 = xF5*xF1
        xF4 = 1/((1/xF2)+(1/xF6))

        # 16: Xf (dimensionless) (inactivation in i_f)
        alpha_Xf = xF1*exp((v_m)/xF2)
        beta_Xf = xF3*exp((v_m)/xF4)
        Xf_inf = alpha_Xf/(alpha_Xf+beta_Xf)
        tau_Xf = ((1./(alpha_Xf+beta_Xf))+xF_const)
        d_Xf = (Xf_inf-Xf)/tau_Xf

        # Current:
        g_f = self.x_F[0]*g_f
        # nS_per_pF (in i_f)
        NatoK_ratio = .491  # Verkerk et al. 2013
        Na_frac = NatoK_ratio/(NatoK_ratio+1)
        i_fNa = Na_frac*g_f*Xf*(v_m-E_Na)
        i_fK = (1-Na_frac)*g_f*Xf*(v_m-E_K)
        i_f = i_fNa+i_fK

        return [d_Xf, i_f, i_fNa, i_fK]

    def i_NaCa(self, v_m, Cai, Nai, k_NaCa):
        # Na+/Ca2+ Exchanger current (INaCa):
        # Ten Tusscher formulation
        KmCa = 1.38    # Cai half-saturation constant millimolar (in i_NaCa)
        KmNai = 87.5    # Nai half-saturation constnat millimolar (in i_NaCa)
        Ksat = 0.1    # saturation factor dimensionless (in i_NaCa)
        gamma = 0.35*2    # voltage dependence parameter dimensionless (in i_NaCa)
        # factor to enhance outward nature of inaca dimensionless (in i_NaCa)
        alpha = 2.5*1.1
        # maximal inaca pA_per_pF (in i_NaCa)
        kNaCa = 1000*1.1*k_NaCa
        i_NaCa = kNaCa*((exp(gamma*v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin))*(Nai ** 3.0)*self.Cao)-(exp(
            (gamma-1.0)*v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin))*(
            self.Nao ** 3.0)*Cai*alpha))/(((KmNai ** 3.0)+(self.Nao ** 3.0))*(KmCa+self.Cao)*(
                1.0+Ksat*exp((gamma-1.0)*v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin))))

        return i_NaCa

    def i_NaK(self, v_m, Nai, p_NaK):
        Km_K = 1.0    # Ko half-saturation constant millimolar (in i_NaK)
        Km_Na = 40.0  # Nai half-saturation constant millimolar (in i_NaK)
        # maxiaml nak pA_per_pF (in i_NaK)
        PNaK = 1.362*1.818*p_NaK
        i_NaK = PNaK*((self.Ko*Nai)/((self.Ko+Km_K)*(Nai+Km_Na)*(1.0 + 0.1245*exp(
            -0.1*v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin))+0.0353*exp(-v_m*self.f_coulomb_per_mmole/(self.r_joule_per_mole_kelvin*self.t_kelvin)))))

        return i_NaK

    def i_up(self, Cai, v_max_up):
        # SR Uptake/SERCA (J_up):
        # Ten Tusscher formulation
        Kup = 0.00025*0.702    # millimolar (in calcium_dynamics)
        # millimolar_per_milisecond (in calcium_dynamics)
        VmaxUp = 0.000425 * 0.26 * v_max_up
        i_up = VmaxUp/(1.0+Kup ** 2.0/Cai ** 2.0)

        return i_up

    def i_leak(self, Ca_SR, Cai, V_leak):
        # SR Leak (J_leak):
        # Ten Tusscher formulation
        V_leak = V_leak*0.00008*0.02
        i_leak = (Ca_SR-Cai)*V_leak

        return i_leak

    def i_rel(self, Ca_SR, Cai, R, O, I, ks):
        ks = 12.5*ks  # [1/ms]
        koCa = 56320*11.43025              # [mM**-2 1/ms]
        kiCa = 54*0.3425                   # [1/mM/ms]
        kom = 1.5*0.1429                   # [1/ms]
        kim = 0.001*0.5571                 # [1/ms]
        ec50SR = 0.45
        MaxSR = 15
        MinSR = 1

        kCaSR = MaxSR - (MaxSR-MinSR)/(1+(ec50SR/Ca_SR)**2.5)
        koSRCa = koCa/kCaSR
        kiSRCa = kiCa*kCaSR
        RI = 1-R-O-I

        d_R = (kim*RI-kiSRCa*Cai*R) - (koSRCa*Cai**2*R-kom*O)
        d_O = (koSRCa*Cai**2*R-kom*O) - (kiSRCa*Cai*O-kim*I)
        d_I = (kiSRCa*Cai*O-kim*I) - (kom*I-koSRCa*Cai**2*RI)

        i_rel = ks*O*(Ca_SR-Cai)*(self.V_SR/self.Vc)

        return [d_R, d_O, d_I, i_rel]

    def i_b_Na(self, v_m, E_Na, g_b_Na):
        g_b_Na = .00029*1.5*g_b_Na    # nS_per_pF (in i_b_Na)
        i_b_Na = g_b_Na*(v_m-E_Na)

        return i_b_Na

    def i_b_Ca(self, v_m, E_Ca, g_b_Ca):
        g_b_Ca = .000592*0.62*g_b_Ca    # nS_per_pF (in i_b_Ca)
        i_b_Ca = g_b_Ca*(v_m-E_Ca)

        return i_b_Ca

    def i_PCa(self, Cai, g_PCa): # SL Pump
        g_PCa = 0.025*10.5*g_PCa    # pA_per_pF (in i_PCa)
        KPCa = 0.0005    # millimolar (in i_PCa)
        i_PCa = g_PCa*Cai/(Cai+KPCa)

        return i_PCa

    def Ca_SR_conc(self, Ca_SR, i_up, i_rel, i_leak):
        # 2: CaSR (millimolar)
        # rapid equilibrium approximation equations --
        # not as formulated in ten Tusscher 2004 text
        Buf_SR = 10.0*1.2  # millimolar (in calcium_dynamics)
        Kbuf_SR = 0.3  # millimolar (in calcium_dynamics)
        Ca_SR_bufSR = 1/(1.0+Buf_SR*Kbuf_SR/(Ca_SR+Kbuf_SR)**2.0)

        d_Ca_SR = Ca_SR_bufSR*self.Vc/self.V_SR*(i_up-(i_rel+i_leak))

        return d_Ca_SR

    def Cai_conc(self, Cai, i_leak, i_up, i_rel, i_CaL_Ca, i_CaT, 
                 i_b_Ca, i_PCa, i_NaCa, Cm):
        # 3: Cai (millimolar)
        # rapid equilibrium approximation equations --
        # not as formulated in ten Tusscher 2004 text
        Buf_C = .06  # millimolar (in calcium_dynamics)
        Kbuf_C = .0006  # millimolar (in calcium_dynamics)
        Cai_bufc = 1/(1.0+Buf_C*Kbuf_C/(Cai+Kbuf_C)**2.0)


        d_Cai = (Cai_bufc)*(i_leak-i_up+i_rel - 
                (i_CaL_Ca+i_CaT+i_b_Ca+i_PCa-2*i_NaCa)*Cm/(2.0*self.Vc*self.f_coulomb_per_mmole))

        return d_Cai

    def Nai_conc(self, i_Na, i_b_Na, i_fNa, i_NaK, i_NaCa, i_CaL_Na, Cm, t):
        # 4: Nai (millimolar) (in sodium_dynamics)
        d_Nai = -Cm*(i_Na+i_b_Na+i_fNa+3.0*i_NaK+3.0*i_NaCa + 
                    i_CaL_Na)/(self.f_coulomb_per_mmole*self.Vc)

        return d_Nai

    def Ki_conc(self, i_K1, i_to, i_Kr, i_Ks, i_fK, i_NaK, i_CaL_K, Cm):
        d_Ki = -Cm*(i_K1+i_to+i_Kr+i_Ks+i_fK - 2.*i_NaK + 
                    i_CaL_K)/(self.f_coulomb_per_mmole*self.Vc)

        return d_Ki


class Ishi():
    Mg_in = 1
    SPM_in = 0.005
    phi = 0.9

    def __init__(self):
        pass

    @classmethod
    def I_K1(cls, V, E_K, y1, K_out, g_K1):
        IK1_alpha = (0.17*exp(-0.07*((V-E_K) + 8*cls.Mg_in)))/(1+0.01*exp(0.12*(V-E_K)+8*cls.Mg_in))
        IK1_beta = (cls.SPM_in*280*exp(0.15*(V-E_K)+8*cls.Mg_in))/(1+0.01*exp(0.13*(V-E_K)+8*cls.Mg_in));
        Kd_spm_l = 0.04*exp(-(V-E_K)/9.1);
        Kd_mg = 0.45*exp(-(V-E_K)/20);
        fo = 1/(1 + (cls.Mg_in/Kd_mg));
        y2 = 1/(1 + cls.SPM_in/Kd_spm_l);
        
        d_y1 = (IK1_alpha*(1-y1) - IK1_beta*fo**3*y1);

        gK1 = 2.5*(K_out/5.4)**.4 * g_K1 
        I_K1 = gK1*(V-E_K)*(cls.phi*fo*y1 + (1-cls.phi)*y2);

        return [I_K1, y1]


class SimplifiedExperimentalArtefacts():
    """
    Experimental artefacts from Lei 2020
    For a cell model that includes experimental artefacts, you need to track
    three additional differential parameters: 

    The undetermined variables are: v_off, g_leak, e_leak
    Given the simplified model in section 4c, 
    you can make assumptions that allow you to reduce the undetermined
    variables to only:
        v_off_dagger – mostly from liquid-junction potential
        g_leak_dagger
        e_leak_dagger (should be zero)
    """
    def __init__(self, r_pipette, c_m, alpha):
        """
        Parameters:
            Experimental measures:
                r_pipette – series resistance of the pipette
                c_m – capacitance of the membrane
            Clamp settings
                alpha – requested proportion of series resistance compensation
        """
        self.r_pipette = r_pipette
        self.c_m = c_m
        self.alpha = alpha

    def dvm_dt(self, g_leak, e_leak, v_off, i_ion, v_cmd, v_m):
        i_leak = self.get_i_leak(g_leak, v_m, e_leak)
        i_out = i_ion + i_leak

        v_p = self.get_v_pipette(v_cmd, i_out)

        dvm_dt = ((1/(self.r_pipette*self.c_m))*(v_p + v_off-v_m) -
                  i_out/self.c_m)

        return dvm_dt

    def get_i_leak(self, g_leak, e_leak, v_m):
        return g_leak * (v_m - e_leak)

    def get_v_pipette(self, v_cmd, i_out):
        v_p = v_cmd + self.alpha * self.r_pipette * i_out

        return v_p


class ExperimentalArtefacts():
    """
    Experimental artefacts from Lei 2020
    For a cell model that includes experimental artefacts, you need to track
    three additional differential parameters: 

    The undetermined variables are: v_off, g_leak, e_leak
    Given the simplified model in section 4c,
    you can make assumptions that allow you to reduce the undetermined
    variables to only:
        v_off_dagger – mostly from liquid-junction potential
        g_leak_dagger
        e_leak_dagger (should be zero)
    """
    def __init__(self, alpha=.85, tau_clamp=.8E-3, tau_sum=20E-3, 
                 tau_z=7.5E-3, c_p_star=4, c_m_star=60):
        """
        Parameters:
            Experimental measures:
                r_pipette – series resistance of the pipette
                c_m – capacitance of the membrane
            Clamp settings
                alpha – requested proportion of series resistance compensation
        """
        self.alpha = alpha
        self.tau_clamp = tau_clamp
        self.tau_sum = tau_sum
        self.tau_z = tau_z
        self.c_p_star = c_p_star
        self.c_m_star = c_m_star

    def get_dvm_dt_simple(self, r_s_star, c_m_star, v_p, v_off, v_m, i_out):
        return ((1/(r_s_star*c_m_star))*(v_p + v_off - v_m) -
                (1/c_m_star) * i_out)

    def get_dvm_dt(self, c_m, v_off, r_s, v_p, v_m, i_ion, i_leak):
        dvm_dt = ((1/(r_s*c_m)) * (v_p + v_off - v_m) - (1/c_m) *
                  (i_ion + i_leak))

        return dvm_dt

    def get_i_leak(self, g_leak, e_leak, v_m):
        return g_leak * (v_m - e_leak)
    
    def get_dvp_dt(self, v_clamp, v_p):
        return (1/self.tau_clamp)*(v_clamp - v_p)
    
    def get_dvclamp_dt(self, v_cmd, r_s_star, i_out, v_clamp):
        return ((1/self.tau_sum)*((v_cmd + self.alpha *
                              r_s_star * i_out) - v_clamp))

    def get_i_in(self, i_ion, i_leak, c_p, dvp_dt, dvclamp_dt,
                 c_m, dvm_dt):
        i_cp = c_p * dvp_dt - self.c_p_star * dvclamp_dt
        i_cm = c_m * dvm_dt - self.c_m_star * dvclamp_dt

        i_in = i_ion + i_leak + i_cp + i_cm

        return i_in, i_cp, i_cm

    def get_diout_dt(self, i_in, i_out):
        return (1/self.tau_z) * (i_in - i_out)


    def get_v_pipette(self, v_cmd, i_out):
        v_p = v_cmd + self.alpha * self.r_pipette * i_out

        return v_p





def kernik_model_inputs():
    return np.array([
        1.000000000000000000e+00, 1.000000000000000000e+00
        ,1.000000000000000000e+00 ,1.000000000000000000e+00
        ,1.000000000000000000e+00 ,1.000000000000000000e+00
        ,1.000000000000000000e+00 ,1.000000000000000000e+00
        ,1.000000000000000000e+00 ,1.000000000000000000e+00
        ,1.000000000000000000e+00 ,1.000000000000000000e+00
        ,1.000000000000000000e+00 ,1.000000000000000000e+00
        ,1.000000000000000000e+00 ,1.000000000000000000e+00
        ,1.337857777976060036e-01 ,4.779949722170410142e-01
        ,2.724275587934869947e+01 ,4.925023317814122947e+00
        ,8.722237600068819319e+00 ,5.663619749982441931e+01
        ,2.180249999999999966e-01 ,5.748852374350000259e-03
        ,1.362349263625756812e+01 ,4.763057118183600114e-02
        ,-7.068087429655488307e+00 ,1.245664052682700015e-02
        ,-2.599445816443767399e+01 ,3.734263315010406359e+01
        ,2.209196423539016507e+01 ,5.000000000000000000e+01
        ,0.000000000000000000e+00 ,7.700000000000000247e-03
        ,1.165584479999999925e-03 ,6.672683867589361034e+04
        ,2.804589082499999719e-01 ,-1.886697157290999982e+01
        ,4.741149999999999491e-06 ,1.178333333332999971e-01
        ,5.536141817130000448e-02 ,1.168420234296690019e+01
        ,3.989181080377499633e+00 ,-1.104713930120320065e+01
        ,3.442309443000000132e-04 ,-1.763447228980960091e+01
        ,1.867605369096950199e+02 ,8.180933873322700833e+00
        ,6.967584211714999975e-01 ,1.122445772394689989e+01
        ,3.080276913789999904e-01 ,1.296629418972199943e+01
        ,7.079145964710999550e+00 ,4.490941550699999868e-02
        ,-6.909880369241999887e+00 ,5.125898259999999871e-04
        ,-4.950571203386999741e+01 ,1.931211223514319045e+03
        ,5.730027499698999272e+00 ,1.658246946830000068e+00
        ,1.004625591711029955e+02 ,1.849999999999999978e-01
        ,9.720613409241000369e+00 ,1.080458463848179917e+02
        ,1.310701573394099917e+01 ,2.326914366999999640e-03
        ,-7.917726289513000282e+00 ,3.626598863999999992e-03
        ,-1.983935886002599958e+01 ,9.663294977114741414e+03
        ,7.395503564612999625e+00 ,5.122571819999999351e-04
        ,-6.658375550265199649e+01 ,3.197758038399999697e-02
        ,1.673315025160000136e-01 ,9.510887249620000317e-01
        ,4.349999999999999700e-02 ,5.789700000000000023e-07
        ,-1.458971217019999855e+01 ,2.008665023788437247e+04
        ,1.020235284528000186e+01 ,2.394529134652999858e+01
        ,0.000000000000000000e+00 ,0.000000000000000000e+00
        ,0.000000000000000000e+00 ,0.000000000000000000e+00
        ,0.000000000000000000e+00])
