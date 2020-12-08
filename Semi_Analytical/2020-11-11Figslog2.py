import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from math import floor, log10
import progressbar

def single_simple_delta(t, argsb ):
    """Emissionsfunktion eines einzelnen Units:
    t           Zeitarray
    a           Anfangsemission
    b           Einsparung pro  Jahr
    lifespan    Lebensdauer
    delta       Breite delta-fkt"""
    a = args['a']
    b = args['b']
    lifespan = args['lifespan']
    delta = args['delta']
    dirac = np.exp(-t**2/delta**2)/(np.sqrt(np.pi)*delta)
    dirac2 = np.exp(-(t-lifespan)**2/delta**2)/(np.sqrt(np.pi)*delta)
    SEF = dirac + b*np.heaviside(t,0)*np.heaviside(- t + lifespan, 0)
    SEF2 = dirac2 + b*np.heaviside((t-lifespan),0)*np.heaviside(- (t-lifespan) + lifespan, 0)
    return  SEF + SEF2

def single_simple_block(t, args):
    """Emissionsfunktion eines einzelnen Units:
    t           Zeitarray
    a           Anfangsemission
    b           Einsparung pro  Jahr
    lifespan    Lebensdauer"""
    a = args['a']
    b = args['b']
    lifespan = args['lifespan']
    breite = args['breite']
    dx = (t[-1]-t[0])/(len(t)-1)
    y = b*np.heaviside(t,0)*np.heaviside(- t + lifespan, 0)
    for i in np.arange(len(t)):
        if t[i]>0:
            for m in np.arange(int(breite)):
                y[i+m] += a/(breite*dx)
            break
    return y

def strommix(t, args):
    """Strommix über Zeit
    Annahme: linearer Abfall an CO_2 Emissionen pro kWh
    Momentan: 400g pro kWh im Jahr 2020
    neutral         Jahr in dem Dtl 100% Ökostrom hat
    return kg pro kWh zur zeit t
    """
    neutral = args['neutral']
    pa = -0.401/(neutral-2019)                     # pro Jahr
    erg = 0.401+pa*(t+1)
    return erg.clip(min=0)


def single_simple_real(t,year, args):
    """
    alles wird in kg gerechnet
    a                           Anfangsemission
    b                           Ausstoss Verbrenner pro Jahr
    """
    lifekm, driving, breite, neu_ausstoß, neu_ausstoß_ICE, verbrauch, recycling , ausstoss_ICE = args['lifekm'], args['driving'], args['breite'], args['neu_ausstoß'], args['neu_ausstoß_ICE'], args['verbrauch'], args['recycling'], args['ausstoss_ICE']
    lifetime = lifekm/driving
    a = neu_ausstoß-neu_ausstoß_ICE
    dx = (t[-1]-t[0])/(len(t)-1)
    np.ceil((t[-1]-t[0])/lifetime)

    y = np.zeros_like(t)
    for i in np.arange(np.ceil((t[-1]-t[0])/lifetime)):
        einsparung = driving*verbrauch*strommix(t+ year, args)-ausstoss_ICE*driving
        y += einsparung*np.heaviside(t-i*lifetime,0)*np.heaviside(- (t-lifetime*i) + lifetime, 0)
        j = np.searchsorted((t-i*lifetime),0,side='left')
        if j<len(t):
            y[j] += a/dx - breite*a
            for m in np.arange(int(breite)):
                y[j-m] += a
    return y

"""ab hier könnte auch alles importiert werden, weiß gerade nur nicht wie das geht"""
def faltung2(v, f, t):
    """Faltung der Net emission of a single unit mit growth fkt
    f       net emission of a single Unit
    v       growth funktion
    """
    dx = (t[-1]-t[0])/(len(t)-1)
    g = np.zeros(len(t))
    # herausfinden wo t = 0
    for i in np.arange(len(t)):
        if t[i] >= 0:
            t_0 = i
            break
    # Faltung ab t = 0
    for i in np.arange(len(t)):
        g[0+i]= np.sum(v[0:i+1]*np.flip(f[0:i+1])*dx)
    return g


def faltung(growth, NSEF, t, args):
    dx = (t[-1]-t[0])/(len(t)-1)
    g = np.zeros(2*len(t))
    v=growth(t-t0,args)
    for i in np.arange(len(t)):
        f = NSEF(t,t[i]-t0,args)
        g[(i):(i+len(t))]+= v[i]*f*dx
    return g[:len(t)]

def growth_exp(t, args):
    """Wachstumsfunktion Zubau; hier Annahme exponentiell
    t           Zeitenarray
    r           Exp. Rate"""
    r, a_0 = args['r'], args['a_0']
    a = np.zeros(len(t))
    # for i in np.arange(len(t)):
    #     if t[i]>0:
    #         a[i:] = a_0*r*np.exp(r*t[i:])
    #         break
    for i in np.arange(len(t)):
        a[i:] = a_0*r*np.exp(r*t[i:])
    return a

def growth_lin(t, args):
    """Wachstumsfunktion Zubau; hier Annahme linear
    t           Zeitenarray
    c           Zubaurate"""
    c = args['c']
    a = np.zeros(len(t))
    for i in np.arange(len(t)):
        #if t[i]>0:
        #    a[i:] = c
        #    break
        a[i:] = c
    return a

def growth_log_int(t, args):
    """Wachstumsfunktion Zubau; hier Annahme logistisch
    t           Zeitenarray
    r           Exp. Rate"""
    k, xhalb, amax = args['k'], args['xhalb'], args['amax']
    return amax/(1+np.exp(-k*(t-xhalb)))

    a = np.zeros(len(t))
    for i in np.arange(len(t)):
        if t[i]>0:
            a[i:] = a_0*amax/(a_0+(amax-a_0)*np.exp(-amax*k*t[i:]))
            break
    return a

def growth_log(t, args):
    """Wachstumsfunktion Zubau; hier Annahme logistisch
    t           Zeitenarray
    r           Exp. Rate"""
    k,xhalb, amax, a_0 = args['k'], args['xhalb'], args['amax'], args['a_0']


    a = np.zeros(len(t))
    a = amax*(amax/a_0-1)*amax*k*np.exp(-amax*k*t)/(((amax/a_0-1)*np.exp(-amax*k*t)+1))**2
    return a

def amor_time(t, array):
    """Amortization time fuer beliebiges Array ausrechnen
    t       Zeitenarray
    Array   Wertearray"""
    for i in np.arange(len(array)):
        if array[i]<0:
            return t[i]
            break
    return(0)

def round_sig(x, sig=2):
    return round(x, sig-int(floor(log10(abs(x))))-1)

def func(label):
    """RadioButton"""
    hzdict = {'1': 1, '2': 2, '3': 3,'4': 4,'5': 5,'6': 6,'7': 7,'8': 8,'9': 9,'10': 10}
    global which
    which = hzdict[label]

def main():
    args = {}
    args['a'] = 'FEHLER1'                       # Anfangsemission
    args['b'] = 'FEHLER2'                      # Ausstoss Verbrenner pro Jahr
    args['lifespan'] = 'FEHLER3'               # Lebensdauer
    args['delta'] = 0.1                 # breite der Deltafunktion
    args['n'] = 10000

    # REALISTISCH
    args['lifekm'] = 230000             # Lebensdauer in km
    args['driving'] = 14000             # gefahrene Kilometer pro Jahr
    args['breite'] = 0 #args['n']/100
    args['neutral'] = 2050
    args['neu_ausstoß'] = 11600          # kgCO_2-Ausstoß EV
    args['neu_ausstoß_ICE'] = 6000      # kgCO_2-Ausstoß ICE
    args['verbrauch'] = 0.165            # kWh pro km
    args['ausstoss_ICE'] = 0.134          # kg pro km
    args['recycling'] = 0             # kg CO_2 für recycling

    # logistisches Wachstum
    args['k'] = 9.38342418*10**(-9)                    # r-Wert für logistisches Wachstum
    args['xhalb'] = 2050
    args['amax'] = 47.7*10**6               # Endwert für log Wachstum
    args['log_0'] = 688.317359                         # Anfangszahl von e-Autos bei exp. growth
    global t0
    t0 = -2


    # Aus vorherigem Dokument
    args['a_0'] = 136617                     # Anfangszahl von e-Autos bei exp. growth
    args['r'] = 0.447                     # growth rate
    args['c'] = 1                       # Zubaurate Units pro Zeit - lin

    # Grundgerüst
    t = np.linspace(t0,50,args['n'])    # Zeitenarray
    dx = (t[-1]-t[0])/(len(t)-1)

    scenario = 1
    which = int(input(' 0 = SEF \n 1 = FEHLER \n 2 = nur real faktor \n 3 = Vergleich Faktoren \n 4 = Suszeptibilities'))
     # 1=single, 2= Int, 3=growth, 4=Faltung, 5=total emmissions, 6= r-abh., 7 =k-abh, 8=lifetime abhgk., 9 = neuaussto abhgk., 10=neutral jahr,

    fkt = single_simple_real   #_delta, _block
    growth = growth_log         #_lin, _exp, log
    args['a_00']=args['a_0']
    NSEF = fkt

    bar = progressbar.ProgressBar(maxval=100, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    if which == 0:
        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(1, 1, 1)
        werte = NSEF(t, 0, args)
        fig.suptitle('Net Single Emission Rate f(t)', size = 25)            # Titel fuer Plots
        ax.set_xlim(t[0]+2020, t[-1]+2020)                            # Achsenlimits x-Achse
        ax.set_ylim(round_sig(1.2*np.min(werte)/1000), round_sig(1.2*(args['neu_ausstoß']-args['neu_ausstoß_ICE'])/1000))                             # Achsenlimits y-Achse
        ax.set_ylabel(r'Net $CO_2$ emission rate in $\frac{tons\ CO_2}{year}$', size = 20)   # Beschriftung y-Achse
        ax.set_xlabel(r'Time $t$ in years', size=20)                # Beschriftung x-Achse
        ax.plot(t+2020, werte/1000)
        plt.xticks(size=17)
        plt.yticks(size=17)
        plt.savefig("/Users/diez/X-Uni/2020_0SoSe/BA/Finalfigs/06NSEFlog.pdf")


    if which == 1:
        """Logistic"""
        size=10
        sizetitle=13
        b=2020
        xticks = np.arange(b, 31+b, 10)
        fig, ((ax1, ax2, ax3, ax4),(ax5, ax6, ax7, ax8),(ax9, ax10, ax11, ax12)) = plt.subplots(3,4)
        plt.subplots_adjust(hspace=0.4, wspace=0.4)

        v = growth(t, args)
        ax1.set_xticks(xticks)
        ax1.set_title('Growth functions\n'r'$\nu_{log} (t)$', size=sizetitle)            # Titel fuer Plots
        #fig.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax1.set_xlim(t[0]+b, 30+b)                            # Achsenlimits x-Achse
        ax1.set_ylim(0, round_sig(1.2*np.max(v)))
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax1.set_ylabel('$c_{log} = 9.38\cdot 10^{-9}$ (realistic) \n \n 'r'$\nu(t)$ in years$^{-1}$', size=size)   # Beschriftung y-Achse
        ax1.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax1.plot(t+b, v)

        args['r'] = 0.447
        args['a_0'] = args['a_00']*0.447/args['r']
        v = np.cumsum(growth(t, args)*dx)
        ax2.set_xticks(xticks)
        ax2.set_title('Total amount of EVs\n'r'$N_{tot}^{log}(t)$', size=sizetitle)            # Titel fuer Plots
        #fg.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax2.set_xlim(t[0]+b, 30+b)                            # Achsenlimits x-Achse
        ax2.set_ylim(0, 80*10**6)
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax2.set_ylabel(r''r'$\nu(t)$ in $\frac{1}{year}', size=size)   # Beschriftung y-Achse
        ax2.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax2.hlines(47.7*10**6,b, b+50, 'r', ls='--')
        # ax2.vlines(2033.1, 0, 47.7*10**6, 'r', ls='--')
        ax2.plot(t+b, v)

        #f = fkt(t,args)
        v = growth(t,args)
        g = faltung(growth,NSEF,t,args)
        ax3.set_title('Net emission rate g(t)\nConvolution', size = sizetitle)            # Titel fuer Plots
        ax3.set_xlim(t[0]+2020, 2050)                            # Achsenlimits x-Achse
        ax3.set_ylim(round_sig(-1.2*np.max(g/1000)), round_sig(1.2*np.max(g/1000)))
        ax3.set_xticks(xticks)

        #round_sig(np.max(g)/1000*1.2))
                                    # Achsenlimits y-Achse
        ax3.set_ylabel(r'$CO_2$ emission in $\frac{t}{year}$', size = size)   # Beschriftung y-Achse
        ax3.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax3.plot(t+2020, g/1000)
        ax3.axhline(y=0, c='k', ls='--')


        G = dx*np.cumsum(g)
        ax4.set_title('Net total emissions \nG(t)', size=sizetitle)            # Titel fuer Plots
        ax4.set_xlim(t[0]+2020, 2050)                            # Achsenlimits x-Achse
        ax4.set_ylim(round_sig(-1.2*np.max(G/1000)), round_sig(1.2*np.max(G/1000)))                             # Achsenlimits y-Achse
        ax4.set_ylabel(r'Total $CO_2$ emission in t', size = size)   # Beschriftung y-Achse
        ax4.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax4.plot(t+2020, G/1000)
        ax4.set_xticks(xticks)
        ax4.axhline(y=0, c='k', ls='--')


        v = growth(t, args)
        ax5.set_xticks(xticks)
        #fig.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax5.set_xlim(t[0]+b, 30+b)                            # Achsenlimits x-Achse
        ax5.set_ylim(0, 50*10**5)
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax5.set_ylabel('$c_{exp} = 0.17$ \n \n 'r'$\nu(t)$ in years$^{-1}$', size=size)   # Beschriftung y-Achse
        ax5.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax5.plot(t+b, v)


        v = np.cumsum(growth(t, args)*dx)
        ax5.set_xticks(xticks)
        #ax5.set_title('Total amount of EVs with $c_{exp} = 0.17$', size=sizetitle)            # Titel fuer Plots
        #fig.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax6.set_xlim(t[0]+b, 30+b)                            # Achsenlimits x-Achse
        ax6.set_ylim(0, 80*10**6)
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax6.set_ylabel(''r'$\nu(t)$ in years$^{-1}$', size=size)   # Beschriftung y-Achse
        ax6.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax6.plot(t+b, v)
        ax6.hlines(47.7*10**6,b, b + 28.75732428804782, 'r', ls='--')
        ax6.vlines(b + 28.75732428804782, 0, 47.7*10**6, 'r', ls='--')

        #f = fkt(t,args)
        v = growth(t,args)
        g = faltung(growth,NSEF,t,args)
        ax7.set_yticks(np.linspace(-1*10**6,1*10**6,11))

        ax7.set_ylim(- 5*10**5, 5*10**5)
        ax7.set_xlim(t[0]+2020, 2050)                            # Achsenlimits x-Achse
        ax7.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax7.set_xticks(xticks)

        #round_sig(np.max(g)/1000*1.2))
                                    # Achsenlimits y-Achse
        ax7.set_ylabel(r'$CO_2$ emission in $\frac{t}{year}$', size = size)   # Beschriftung y-Achse
        ax7.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax7.plot(t+2020, g/1000)
        ax7.axhline(y=0, c='k', ls='--')

        G = dx*np.cumsum(g)
        ax8.set_xlim(t[0]+2020, 2050)                            # Achsenlimits x-Achse
        ax8.set_ylim(-5*10**6, 5*10**6)                             # Achsenlimits y-Achse
        ax8.set_ylabel(r'Total $CO_2$ emission in t', size = size)   # Beschriftung y-Achse
        ax8.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax8.plot(t+2020, G/1000)
        ax8.set_xticks(xticks)
        ax8.axhline(y=0, c='k', ls='--')
    if which ==2:
        """Exponential"""
        size=10
        sizetitle=13
        b=2020
        xticks = np.arange(b, 51+b, 10)
        xmax = 50
        fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1,4)
        plt.subplots_adjust(hspace=0.4, wspace=0.4)

        v = growth(t, args)
        ax1.set_xticks(xticks)
        ax1.set_title('Growth functions\n'r'$\nu_{log} (t)$', size=sizetitle)            # Titel fuer Plots
        #fig.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax1.set_xlim(t[0]+b, xmax+b)                            # Achsenlimits x-Achse
        ax1.set_ylim(0, round_sig(1.2*np.max(v)))
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax1.set_ylabel('$c_{log} = 9.38\cdot 10^{-9}$ (realistic) \n \n 'r'$\nu(t)$ in years$^{-1}$', size=size)   # Beschriftung y-Achse
        ax1.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax1.plot(t+b, v)

        args['r'] = 0.447
        args['a_0'] = args['a_00']*0.447/args['r']
        v = np.cumsum(growth(t, args)*dx)
        ax2.set_xticks(xticks)
        ax2.set_title('Total amount of EVs\n'r'$N_{tot}^{log}(t)$', size=sizetitle)            # Titel fuer Plots
        #fg.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax2.set_xlim(t[0]+b, xmax+b)                            # Achsenlimits x-Achse
        ax2.set_ylim(0, 80*10**6)
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax2.set_ylabel(r''r'$\nu(t)$ in years$^{-1}$', size=size)   # Beschriftung y-Achse
        ax2.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax2.hlines(47.7*10**6,b, b+50, 'r', ls='--')
        # ax2.vlines(2033.1, 0, 47.7*10**6, 'r', ls='--')
        ax2.plot(t+b, v)

        f = fkt(t,args)
        v = growth(t,args)
        g = faltung(growth,f,t)
        G = dx*np.cumsum(g)

        ax3.set_title('Current emission function g(t)\nConvolution', size = sizetitle)            # Titel fuer Plots
        ax3.set_xlim(t[0]+2020, xmax+b)                            # Achsenlimits x-Achse
        ax3.set_ylim(round_sig(-1.2*np.max(G/1000)), round_sig(1.2*np.max(G/1000)))
        ax3.set_xticks(xticks)

        #round_sig(np.max(g)/1000*1.2))
                                    # Achsenlimits y-Achse
        ax3.set_ylabel(r'$CO_2$ emission in $\frac{t}{year}$', size = size)   # Beschriftung y-Achse
        ax3.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax3.plot(t+2020, g/1000)
        ax3.axhline(y=0, c='k', ls='--')


        ax4.set_title('Total Emissions \nG(t)', size=sizetitle)            # Titel fuer Plots
        ax4.set_xlim(t[0]+2020, xmax+b)                            # Achsenlimits x-Achse
        ax4.set_ylim(round_sig(-1.2*np.max(G/1000)), round_sig(1.2*np.max(G/1000)))                             # Achsenlimits y-Achse
        ax4.set_ylabel(r'Total $CO_2$ emission in t', size = size)   # Beschriftung y-Achse
        ax4.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax4.plot(t+2020, G/1000)
        ax4.set_xticks(xticks)
        ax4.axhline(y=0, c='k', ls='--')

    if which ==3:
        """Exponential"""
        size=12
        sizetitle=15
        b=2020
        xticks = np.arange(b, 51+b, 20)
        xmax = 40
        fig, ((ax5, ax6, ax7, ax8),(ax1, ax2, ax3, ax4),(ax9, ax10, ax11, ax12)) = plt.subplots(3,4, figsize=(15,8))
        plt.subplots_adjust(hspace=0.4, wspace=0.4)

        v = growth(t, args)
        ax1.set_xticks(xticks)
        ax5.set_title('Growth functions\n'r'$\nu_{log} (t)$', size=sizetitle)            # Titel fuer Plots
        #fig.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax1.set_xlim(t[0]+b, xmax+b)                            # Achsenlimits x-Achse
        ax1.set_ylim(0, round_sig(1.2*np.max(v)))
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax1.set_ylabel('$c_{log} = 9.38\cdot 10^{-9}$ (realistic) \n \n 'r'$\nu(t)$ in 'r'$\frac{1}{year}$', size=size)   # Beschriftung y-Achse
        ax1.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax1.plot(t+b, v)
        bar.update(1/12*100)
        v = np.cumsum(growth(t, args)*dx)
        ax2.set_xticks(xticks)
        ax6.set_title('Total amount of EVs\n'r'$N_{tot}^{log}(t)$', size=sizetitle)            # Titel fuer Plots
        #fg.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax2.set_xlim(t[0]+b, xmax+b)                            # Achsenlimits x-Achse
        ax2.set_ylim(0, 80*10**6)
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax2.set_ylabel(r'$N^{log}_{tot}(t)$', size=size)   # Beschriftung y-Achse
        ax2.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax2.hlines(47.7*10**6,b, b+50, 'r', ls='--')
        # ax2.vlines(2033.1, 0, 47.7*10**6, 'r', ls='--')
        ax2.plot(t+b, v)
        bar.update(1/12*100)

        #f = fkt(t,args)
        v = growth(t,args)
        g = faltung(growth,NSEF,t,args)
        G = dx*np.cumsum(g)

        ax7.set_title('Net emission rate \n g(t)', size = sizetitle)            # Titel fuer Plots
        ax3.set_xlim(t[0]+2020, xmax+b)                            # Achsenlimits x-Achse
        ax3.set_ylim(round_sig(-1.2*np.max(G/1000)), round_sig(1.2*np.max(G/1000)))
        ax3.set_xticks(xticks)
        bar.update(3/12*100)
        #round_sig(np.max(g)/1000*1.2))
                                    # Achsenlimits y-Achse
        ax3.set_ylabel(r'g(t) in $\frac{tons CO_2}{year}$', size = size)   # Beschriftung y-Achse
        ax3.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax3.plot(t+2020, g/1000)
        ax3.axhline(y=0, c='k', ls='--')


        ax8.set_title('Net total emissions \nG(t)', size=sizetitle)            # Titel fuer Plots
        ax4.set_xlim(t[0]+2020, xmax+b)                            # Achsenlimits x-Achse
        ax4.set_ylim(round_sig(-1.2*np.max(G/1000)), round_sig(1.2*np.max(G/1000)))                             # Achsenlimits y-Achse
        ax4.set_ylabel(r'G(t) in tons $CO_2$', size = size)   # Beschriftung y-Achse
        ax4.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax4.plot(t+2020, G/1000)
        ax4.set_xticks(xticks)
        ax4.axhline(y=0, c='k', ls='--')
        bar.update(4/12*100)

        ######################
        args['k'] = 10* 9.38342418*10**(-9)                    # r-Wert für logistisches Wachstum
        gamma = args['amax']/args['a_0'] - 1
        r = args['amax'] * args['k']
        t_halb = 1/r*np.log(gamma)       # args['a_0'] = args['amax']/2 - np.sqrt(args['amax']**2/4-fest[0]/args['k'])                # r-Wert für logistisches Wachstum
        t_halb
        v = growth(t, args)
        ax5.set_xticks(xticks)
        #fig.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax5.set_xlim(t[0]+b, xmax+b)                            # Achsenlimits x-Achse
        ax5.set_ylim(0, round_sig(1.2*np.max(v)))
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax5.set_ylabel('$c_{log} = 9.38\cdot 10^{-8}$ (fast) \n \n 'r'$\nu(t)$ in 'r'$\frac{1}{year}$', size=size)   # Beschriftung y-Achse
        ax5.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax5.plot(t+b, v)
        bar.update(5/12*100)

        v = np.cumsum(growth(t, args)*dx)
        ax6.set_xticks(xticks)
        #fg.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax6.set_xlim(t[0]+b, xmax+b)                            # Achsenlimits x-Achse
        ax6.set_ylim(0, 80*10**6)
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax6.set_ylabel(r'$N^{log}_{tot}(t)$', size=size)   # Beschriftung y-Achse
        ax6.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax6.hlines(47.7*10**6,b, b+50, 'r', ls='--')
        # x2.vlines(2033.1, 0, 47.7*10**6, 'r', ls='--')
        ax6.plot(t+b, v)
        bar.update(6/12*100)


        #f = fkt(t,args)
        v = growth(t,args)
        g = faltung(growth,NSEF,t,args)
        G = dx*np.cumsum(g)

        ax7.set_xlim(t[0]+2020, xmax+b)                            # Achsenlimits x-Achse
        ax7.set_ylim(round_sig(-1.2*np.max(G/1000)), round_sig(1.2*np.max(G/1000)))
        ax7.set_xticks(xticks)

        #round_sig(np.max(g)/1000*1.2))
                                    # Achsenlimits y-Achse
        ax7.set_ylabel(r'g(t) in $\frac{tons CO_2}{year}$', size = size)   # Beschriftung y-Achse
        ax7.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax7.plot(t+2020, g/1000)
        ax7.axhline(y=0, c='k', ls='--')
        bar.update(7/12*100)


        ax8.set_xlim(t[0]+2020, xmax+b)                            # Achsenlimits x-Achse
        ax8.set_ylim(round_sig(-1.2*np.max(G/1000)), round_sig(1.2*np.max(G/1000)))                             # Achsenlimits y-Achse
        ax8.set_ylabel(r'G(t) in tons $CO_2$', size = size)   # Beschriftung y-Achse
        ax8.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax8.plot(t+2020, G/1000)
        ax8.set_xticks(xticks)
        ax8.axhline(y=0, c='k', ls='--')
        bar.update(8/12*100)

        ##########################
        # fest = growth(np.array([0, 1]),args)
        args['k'] = 0.1*9.38342418*10**(-9)
        gamma = args['amax']/args['a_0'] - 1
        r = args['amax'] * args['k']
        t_halb = 1/r*np.log(gamma)       # args['a_0'] = args['amax']/2 - np.sqrt(args['amax']**2/4-fest[0]/args['k'])                # r-Wert für logistisches Wachstum
        print(t_halb)
        # # args['a_0']
        v = growth(t, args)
        401*0.165

        ax9.set_xticks(xticks)
        #fig.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax9.set_xlim(t[0]+b, xmax+b)                            # Achsenlimits x-Achse
        ax9.set_ylim(0, round_sig(1.2*np.max(v)))
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax9.set_ylabel('$c_{log} = 9.38\cdot 10^{-10}$ (slow) \n \n 'r'$\nu(t)$ in 'r'$\frac{1}{year}$', size=size)   # Beschriftung y-Achse
        ax9.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax9.plot(t+b, v)
        ax9.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        bar.update(9/12*100)


        v = np.cumsum(growth(t, args)*dx)
        ax10.set_xticks(xticks)
        #fg.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax10.set_xlim(t[0]+b, xmax+b)                            # Achsenlimits x-Achse
        ax10.set_ylim(0, 80*10**6)
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax10.set_ylabel(r'$N^{log}_{tot}(t)$', size=size)   # Beschriftung y-Achse
        ax10.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax10.hlines(47.7*10**6,b, b+50, 'r', ls='--')
        # ax2.vlines(2033.1, 0, 47.7*10**6, 'r', ls='--')
        ax10.plot(t+b, v)
        bar.update(10/12*100)

        #f = fkt(t,args)
        v = growth(t,args)
        g = faltung(growth,NSEF,t,args)
        G = dx*np.cumsum(g)

        ax11.set_xlim(t[0]+2020, xmax+b)                            # Achsenlimits x-Achse
        ax11.set_ylim(round_sig(-1.2*np.max(G/1000)), round_sig(1.2*np.max(G/1000)))
        ax11.set_xticks(xticks)

        #round_sig(np.max(g)/1000*1.2))
                                    # Achsenlimits y-Achse
        ax11.set_ylabel(r'g(t) in $\frac{tons CO_2}{year}$', size = size)   # Beschriftung y-Achse
        ax11.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax11.plot(t+2020, g/1000)
        ax11.axhline(y=0, c='k', ls='--')
        ax11.ticklabel_format(axis="y", style="sci", scilimits=(0,0))

        bar.update(11/12*100)


        ax12.set_xlim(t[0]+2020, xmax+b)                            # Achsenlimits x-Achse
        ax12.set_ylim(round_sig(-1.2*np.max(G/1000)), round_sig(1.2*np.max(G/1000)))                             # Achsenlimits y-Achse
        ax12.set_ylabel(r'G(t) in tons $CO_2$', size = size)   # Beschriftung y-Achse
        ax12.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax12.plot(t+2020, G/1000)
        ax12.set_xticks(xticks)
        ax12.axhline(y=0, c='k', ls='--')
        ax12.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax8.annotate(r"$\Delta t_{a} = 6.4$ years", xy=(2026.44, 0), xytext=(2036, 1*10**8),arrowprops=dict(arrowstyle="->"), size = 11)
        ax4.annotate(r"$\Delta t_{a} = 15.6$ years", xy=(2035.61, 0), xytext=(2036, 2.5*10**7),arrowprops=dict(arrowstyle="->"), size = 11)
        ax12.annotate(r"$\Delta t_{a} = 10.2$ years", xy=(2030.2, 0), xytext=(2036, 5*10**4),arrowprops=dict(arrowstyle="->"), size = 11)
        ax1.text(0.05, 0.95, 'e)', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax2.text(0.05, 0.95, 'f)', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax3.text(0.05, 0.95, 'g)', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
        ax4.text(0.05, 0.95, 'h)', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)
        ax5.text(0.05, 0.95, 'a)', horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes)
        ax6.text(0.05, 0.95, 'b)', horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes)
        ax7.text(0.05, 0.95, 'c)', horizontalalignment='center', verticalalignment='center', transform=ax7.transAxes)
        ax8.text(0.05, 0.95, 'd)', horizontalalignment='center', verticalalignment='center', transform=ax8.transAxes)
        ax9.text(0.05, 0.95, 'i)', horizontalalignment='center', verticalalignment='center', transform=ax9.transAxes)
        ax10.text(0.05, 0.95, 'j)', horizontalalignment='center', verticalalignment='center', transform=ax10.transAxes)
        ax11.text(0.05, 0.95, 'k)', horizontalalignment='center', verticalalignment='center', transform=ax11.transAxes)
        ax12.text(0.05, 0.95, 'l)', horizontalalignment='center', verticalalignment='center', transform=ax12.transAxes)
        ax1.tick_params(axis='both', which='major', labelsize=13)
        ax2.tick_params(axis='both', which='major', labelsize=13)
        ax3.tick_params(axis='both', which='major', labelsize=13)
        ax4.tick_params(axis='both', which='major', labelsize=13)
        ax5.tick_params(axis='both', which='major', labelsize=13)
        ax6.tick_params(axis='both', which='major', labelsize=13)
        ax7.tick_params(axis='both', which='major', labelsize=13)
        ax8.tick_params(axis='both', which='major', labelsize=13)
        ax9.tick_params(axis='both', which='major', labelsize=13)
        ax10.tick_params(axis='both', which='major', labelsize=13)
        ax11.tick_params(axis='both', which='major', labelsize=13)
        ax12.tick_params(axis='both', which='major', labelsize=13)

        plt.savefig("/Users/diez/X-Uni/2020_0SoSe/BA/Finalfigs/07loggrowth.pdf")
        bar.finish()


    if which == 4:
        fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2, figsize=(15,8))
        plt.subplots_adjust(hspace=0.35, wspace=0.25)

        N1 = 90
        N2 = 10
        O = 100
        P = 60
        Q = 100
        size=10
        sizetitle=13
        b=2020
        xticks = np.arange(b, 51+b, 10)
        yticks = np.arange(b, 21+b, 5)

        xmax = 50

        """k-Abhängigkeit"""
        data = []
        #f = fkt(t,args)
        for i in np.arange(N1):
            args['k'] =(i+10)/50*9.38342418*10**(-9)
            v = growth(t,args)
            g = faltung(growth,NSEF,t,args)
            G = dx*np.cumsum(g)
            gamma = args['amax']/args['a_0'] - 1
            r = args['amax'] * args['k']
            t_halb = 1/r*np.log(gamma)
            data.append([t_halb, amor_time(t,G)])
            bar.update(i)
        for i in np.arange(N2):
            args['k'] =(i+2)*9.38342418*10**(-9)
            v = growth(t,args)
            g = faltung(growth,NSEF,t,args)
            G = dx*np.cumsum(g)
            gamma = args['amax']/args['a_0'] - 1
            r = args['amax'] * args['k']
            t_halb = 1/r*np.log(gamma)
            data.append([t_halb, amor_time(t,G)])
            bar.update(i+90)
        args['k'] =1000*10**(-9)
        v = growth(t,args)
        g = faltung(growth,NSEF,t,args)
        G = dx*np.cumsum(g)
        gamma = args['amax']/args['a_0'] - 1
        r = args['amax'] * args['k']
        t_halb = 1/r*np.log(gamma)
        data.append([t_halb, amor_time(t,G)])
        bar.finish()
        data = np.array(data)
        ax1.set_title('Half value time $t_{1/2}$', size = sizetitle)
        ax1.set_xlim(2020,np.max(data[:,0])+2020)                            # Achsenlimits x-Achse
        ax1.set_ylim(2020, 2040)                             # Achsenlimits y-Achse
        ax1.set_ylabel(r'year of systemic $CO_2$ amortization', size = size)   # Beschriftung y-Achse
        ax1.set_xlabel(r'Half-value time $t_{1/2}$ of transition to electromobility', size = size )                # Beschriftung x-Achse
        ax1.plot(data[:,0]+2020,2020 + data[:,1])
        # ax.vlines(0.35, 0, np.max(data[:,1]), colors='r')
        ax1.vlines(2020+13.076, 0, 2020+15.61, color='r', ls='--')
        print("1Fertig")

        args['k'] = 9.38342418*10**(-9)
        """Lifetime"""
        data = []
        for i in np.arange(O):
            args['lifekm'] = 100000+2000*i
            #f = fkt(t,args)
            v = growth(t,args)
            g = faltung(growth,NSEF,t,args)
            G = dx*np.cumsum(g)
            data.append([args['lifekm'], amor_time(t,G)])
            data
            bar.update(i)
        bar.finish()
        data = np.array(data)
        ax2.set_title('Lifedistance $d_{life}$', size = sizetitle)
        ax2.set_xlim(data[:,0][0]/1000,data[:,0][-1]/1000)                            # Achsenlimits x-Achse
        ax2.set_ylim(2020, 2040)                             # Achsenlimits y-Achse
        ax2.set_ylabel(r'year of systemic $CO_2$ amortization', size = size)   # Beschriftung y-Achse
        ax2.set_xlabel(r'lifedistance $d_{life}$ in 1000 km', size = size )                # Beschriftung x-Achse
        ax2.plot(data[:,0]/1000,2020 + data[:,1])
        # ax.vlines(0.35, 0, np.max(data[:,1]), colors='r')
        ax2.vlines(230, 2020, 2020+15.607, color='r', ls='--')
        ax2.vlines(210, 2020, 2020+15.622, color='k', ls='--')
        ax2.vlines(258, 2020, 2020+15.607, color='k', ls='--')
        args['lifekm'] = 230000             # Lebensdauer in km
          # kg CO_2 für recycling
        print("2Fertig")

        """CO_2-Neutral"""
        data = []
        for i in np.arange(P):
            args['neutral'] = 2021+i
            #f = fkt(t,args)
            v = growth(t,args)
            g = faltung(growth,NSEF,t,args)
            G = dx*np.cumsum(g)
            data.append([args['neutral'], amor_time(t,G)])
            bar.update(i/60*100)
        bar.finish()
        data = np.array(data)
        ax3.set_title('Emission free power generation mix $t_{CN}$', size = sizetitle)
        ax3.set_xlim(data[:,0][0],data[:,0][-1])                            # Achsenlimits x-Achse
        ax3.set_ylim(2020, 2040)                             # Achsenlimits y-Achse
        ax3.set_ylabel(r'year of systemic $CO_2$ amortization', size = size)   # Beschriftung y-Achse
        ax3.set_xlabel(r'year of emission free power generation $t_{CN}$', size = size)                # Beschriftung x-Achse
        ax3.plot(data[:,0],2020 + data[:,1])
        # ax.vlines(0.35, 0, np.max(data[:,1]), colors='r')
        ax3.vlines(2050, 0, 2020+15.61, color='r', ls='--')
        args['neutral'] = 2050
        print("3Fertig")

        "data initial"
        data = []
        for i in np.arange(Q):
            args['neu_ausstoß'] = args['neu_ausstoß_ICE'] + 80*i
            #f = fkt(t,args)
            v = growth(t,args)
            g = faltung(growth,NSEF,t,args)
            G = dx*np.cumsum(g)
            data.append([args['neu_ausstoß']-args['neu_ausstoß_ICE'], amor_time(t,G)])
            bar.update(i)
        bar.finish()
        data = np.array(data)
        ax4.set_title('Net initial emission $a_i$', size=sizetitle )
        ax4.set_xlim(data[:,0][0]/1000,data[:,0][-1]/1000)                            # Achsenlimits x-Achse
        ax4.set_ylim(2020, 2040)                             # Achsenlimits y-Achse
        ax4.set_ylabel(r'year of systemic $CO_2$ amortization', size =size )   # Beschriftung y-Achse
        ax4.set_xlabel(r'Net initial $CO_2$ emission in tons $CO_2$', size = size)                # Beschriftung x-Achse
        ax4.plot(data[:,0]/1000,2020 + data[:,1])
        # ax.vlines(0.35, 0, np.max(data[:,1]), colors='r')
        ax4.vlines(5600, 2020, 2020+15.61, color='r', ls='--')
        ax4.vlines(4500, 2020, 2020+14.087, color='k', ls='--')
        ax4.vlines(6800, 2020, 2020+16.9178, color='k', ls='--')
        ax1.set_yticks(yticks)
        ax2.set_yticks(yticks)
        ax3.set_yticks(yticks)
        ax4.set_yticks(yticks)
        ax1.text(0.05, 0.95, 'a)', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes)
        ax2.text(0.05, 0.95, 'b)', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        ax3.text(0.05, 0.95, 'c)', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes)
        ax4.text(0.05, 0.95, 'd)', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes)

        print("4Fertig")

        plt.savefig("/Users/diez/X-Uni/2020_0SoSe/BA/Finalfigs/08logparams.pdf")
    if which == 5:
        print('Best Case')
        size=10
        sizetitle=13
        b=2020
        xticks = np.arange(b, 51+b, 10)
        xmax = 50
        fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1,4, figsize=(15,8))
        plt.subplots_adjust(hspace=0.4, wspace=0.4)

        args['lifekm'] = 258000             # Lebensdauer in km
        args['neu_ausstoß'] = 11600          # kgCO_2-Ausstoß EV
        args['neu_ausstoß_ICE'] = 7100      # kgCO_2-Ausstoß ICE
        args['verbrauch'] = 0.15            # kWh pro km
        args['ausstoss_ICE'] = 0.1364          # kg pro km

        v = growth(t, args)
        ax1.set_xticks(xticks)
        ax1.set_title('Growth functions\n'r'$\nu_{log} (t)$', size=sizetitle)            # Titel fuer Plots
        #fig.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax1.set_xlim(t[0]+b, xmax+b)                            # Achsenlimits x-Achse
        ax1.set_ylim(0, round_sig(1.2*np.max(v)))
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax1.set_ylabel('$c_{log} = 9.38\cdot 10^{-9}$ (realistic) \n \n Production of EVs in 'r'$\frac{1}{year}$', size=size)   # Beschriftung y-Achse
        ax1.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax1.plot(t+b, v)
        v = np.cumsum(growth(t, args)*dx)
        ax2.set_xticks(xticks)
        ax2.set_title('Total amount of EVs\n'r'$N_{tot}^{log}(t)$', size=sizetitle)            # Titel fuer Plots
        #fg.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax2.set_xlim(t[0]+b, xmax+b)                            # Achsenlimits x-Achse
        ax2.set_ylim(0, 80*10**6)
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax2.set_ylabel(r'$N^{log}_{tot}(t)$', size=size)   # Beschriftung y-Achse
        ax2.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax2.hlines(47.7*10**6,b, b+50, 'r', ls='--')
        # ax2.vlines(2033.1, 0, 47.7*10**6, 'r', ls='--')
        ax2.plot(t+b, v)
        bar.update(1/12*100)

        #f = fkt(t,args)
        v = growth(t,args)
        g = faltung(growth,NSEF,t,args)
        G = dx*np.cumsum(g)

        ax3.set_title('Net emission rate \n g(t)', size = sizetitle)            # Titel fuer Plots
        ax3.set_xlim(t[0]+2020, xmax+b)                            # Achsenlimits x-Achse
        ax3.set_ylim(round_sig(-1.2*np.max(G/1000)), round_sig(1.2*np.max(G/1000)))
        ax3.set_xticks(xticks)
        bar.update(3/12*100)
        #round_sig(np.max(g)/1000*1.2))
                                    # Achsenlimits y-Achse
        ax3.set_ylabel(r'g(t) in $\frac{tons CO_2}{year}$', size = size)   # Beschriftung y-Achse
        ax3.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax3.plot(t+2020, g/1000)
        ax3.axhline(y=0, c='k', ls='--')


        ax4.set_title('Net total emissions \nG(t)', size=sizetitle)            # Titel fuer Plots
        ax4.set_xlim(t[0]+2020, xmax+b)                            # Achsenlimits x-Achse
        ax4.set_ylim(round_sig(-1.2*np.max(G/1000)), round_sig(1.2*np.max(G/1000)))                             # Achsenlimits y-Achse
        ax4.set_ylabel(r'G(t) in tons $CO_2$', size = size)   # Beschriftung y-Achse
        ax4.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax4.plot(t+2020, G/1000)
        ax4.set_xticks(xticks)
        ax4.axhline(y=0, c='k', ls='--')
        bar.update(4/12*100)
    if which == 6:
        print('Worst Case')
        size=10
        sizetitle=13
        b=2020
        xticks = np.arange(b, 51+b, 10)
        xmax = 50
        fig, ((ax1, ax2, ax3, ax4)) = plt.subplots(1,4, figsize=(15,8))
        plt.subplots_adjust(hspace=0.4, wspace=0.4)

        args['lifekm'] = 210000             # Lebensdauer in km
        args['neu_ausstoß'] = 11600          # kgCO_2-Ausstoß EV
        args['neu_ausstoß_ICE'] = 4800      # kgCO_2-Ausstoß ICE
        args['verbrauch'] = 0.18            # kWh pro km
        args['ausstoss_ICE'] = 0.1274          # kg pro km

        v = growth(t, args)
        ax1.set_xticks(xticks)
        ax1.set_title('Growth functions\n'r'$\nu_{log} (t)$', size=sizetitle)            # Titel fuer Plots
        #fig.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax1.set_xlim(t[0]+b, xmax+b)                            # Achsenlimits x-Achse
        ax1.set_ylim(0, round_sig(1.2*np.max(v)))
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax1.set_ylabel('$c_{log} = 9.38\cdot 10^{-9}$ (realistic) \n \n Production of EVs in 'r'$\frac{1}{year}$', size=size)   # Beschriftung y-Achse
        ax1.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax1.plot(t+b, v)
        v = np.cumsum(growth(t, args)*dx)
        ax2.set_xticks(xticks)
        ax2.set_title('Total amount of EVs\n'r'$N_{tot}^{log}(t)$', size=sizetitle)            # Titel fuer Plots
        #fg.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax2.set_xlim(t[0]+b, xmax+b)                            # Achsenlimits x-Achse
        ax2.set_ylim(0, 80*10**6)
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax2.set_ylabel(r'$N^{log}_{tot}(t)$', size=size)   # Beschriftung y-Achse
        ax2.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax2.hlines(47.7*10**6,b, b+50, 'r', ls='--')
        # ax2.vlines(2033.1, 0, 47.7*10**6, 'r', ls='--')
        ax2.plot(t+b, v)
        bar.update(1/12*100)

        #f = fkt(t,args)
        v = growth(t,args)
        g = faltung(growth,NSEF,t,args)
        G = dx*np.cumsum(g)

        ax3.set_title('Net emission rate \n g(t)', size = sizetitle)            # Titel fuer Plots
        ax3.set_xlim(t[0]+2020, xmax+b)                            # Achsenlimits x-Achse
        ax3.set_ylim(round_sig(-1.2*np.max(G/1000)), round_sig(1.2*np.max(G/1000)))
        ax3.set_xticks(xticks)
        bar.update(3/12*100)
        #round_sig(np.max(g)/1000*1.2))
                                    # Achsenlimits y-Achse
        ax3.set_ylabel(r'g(t) in $\frac{tons CO_2}{year}$', size = size)   # Beschriftung y-Achse
        ax3.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax3.plot(t+2020, g/1000)
        ax3.axhline(y=0, c='k', ls='--')

        ax4.set_title('Net total emissions \nG(t)', size=sizetitle)            # Titel fuer Plots
        ax4.set_xlim(t[0]+2020, xmax+b)                            # Achsenlimits x-Achse
        ax4.set_ylim(round_sig(-1.2*np.max(G/1000)), round_sig(1.2*np.max(G/1000)))                             # Achsenlimits y-Achse
        ax4.set_ylabel(r'G(t) in tons $CO_2$', size = size)   # Beschriftung y-Achse
        ax4.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax4.plot(t+2020, G/1000)
        ax4.set_xticks(xticks)
        ax4.axhline(y=0, c='k', ls='--')
        bar.update(4/12*100)

    plt.show()


if __name__ == "__main__":
    main()
