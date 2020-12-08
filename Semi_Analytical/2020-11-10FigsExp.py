import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from math import floor, log10
from matplotlib.ticker import FormatStrFormatter
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
    return  dirac + b*np.heaviside(t,0)*np.heaviside(- t + lifespan, 0)

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
    einsparung = driving*verbrauch*strommix(t+year, args)-ausstoss_ICE*driving


    # Einsparung pro jahr
    y = einsparung*np.heaviside(t,0)*np.heaviside(- t + lifetime, 0)


    i = np.searchsorted(t,0,side='left')
    y[i] += a/dx - breite*a
    for m in np.arange(int(breite)):
        y[i-m] += a

    # j = np.searchsorted(t, lifetime+breite*dx, side='left')
    # y[j] += recycling/dx - breite*recycling
    # for m in np.arange(int(breite)):
    #     y[j+m] += recycling


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
    v = growth(t-t0,args)
    g = np.zeros(2*len(t))
    for i in np.arange(len(t)):
        f = NSEF(t,t[i]-t0,args)
        g[(i):(i+len(t))]+= v[i]*f*dx
    return g[:len(t)]

def growth_exp(t, args):
    """Wachstumsfunktion Zubau; hier Annahme exponentiell
    t           Zeitenarray
    r           Exp. Rate"""
    # for i in np.arange(len(t)):
    #     if t[i] >= 0:
    #         t_0 = i
    #         break
    r, a_0 = args['r'], args['a_0']
    a=np.zeros_like(t)
    i = np.searchsorted(t,0,side='left')
    a[i:] = a_0*np.exp(r*t[i:])
    return a

def growth_exp2(t, args):
    """Wachstumsfunktion Zubau; hier Annahme exponentiell
    t           Zeitenarray
    r           Exp. Rate"""
    for i in np.arange(len(t)):
        if t[i] >= 0:
            t_0 = i
            break
    r, a_0 = args['r'], args['a_0']
    a = a_0*r*np.exp(r*t)
    a[:t_0]=0
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
    k, a_0, amax = args['k'], args['log_0'], args['amax']
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
    k, a_0, amax = args['k'], args['log_0'], args['amax']
    a = np.zeros(len(t))
    for i in np.arange(len(t)):
        if t[i]>0:
            a[i:] = a_0*amax/((a_0+(amax-a_0)*np.exp(-amax*k*t[i:])))**2*(amax-a_0)*(amax)*k*np.exp(-amax*k*t[i:])
            break
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
    args['delta'] = 0.02                 # breite der Deltafunktion
    args['n'] = 10000

    # REALISTISCH
    args['lifekm'] = 230000             # Lebensdauer in km
    args['driving'] = 14000             # gefahrene Kilometer pro Jahr
    args['breite'] = 0#args['n']/400
    args['neutral'] = 2050
    args['neu_ausstoß'] = 11600          # kgCO_2-Ausstoß EV
    args['neu_ausstoß_ICE'] = 6000      # kgCO_2-Ausstoß ICE
    args['verbrauch'] = 0.165            # kWh pro km
    args['ausstoss_ICE'] = 0.1306          # kg pro km
    args['recycling'] = 0             # kg CO_2 für recycling

    # logistisches Wachstum
    args['k'] = 8.79255838*10**(-9)                     # r-Wert für logistisches Wachstum
    args['amax'] = 47.7*10**6               # Endwert für log Wachstum
    args['log_0'] = 688.317359                         # Anfangszahl von e-Autos bei exp. growth


    # Aus vorherigem Dokument
    args['a_0'] = 136617                     # Anfangszahl von e-Autos bei exp. growth
    args['a_00'] = 136617
    args['r'] = 0.75732428804782                     # growth rate
    args['c'] = 1                       # Zubaurate Units pro Zeit - lin

    global t0
    t0 = -2

    # Grundgerüst
    t = np.linspace(t0,100,args['n'])    # Zeitenarray
    dx = (t[-1]-t[0])/(len(t)-1)

    scenario = 1
    which = int(input('1=single, 2= Int, 3=growth, 4=Faltung, 5=total emmissions, 6= r-abh., 7 =k-abh, 8=lifetime abhgk., 9 = neuaussto abhgk., 10=neutral jahr'))
     # 1=single, 2= Int, 3=growth, 4=Faltung, 5=total emmissions, 6= r-abh., 7 =k-abh, 8=lifetime abhgk., 9 = neuaussto abhgk., 10=neutral jahr,

    fkt = single_simple_real   #_delta, _block
    growth = growth_exp2         #_lin, _exp, log
    NSEF = fkt


    bar = progressbar.ProgressBar(maxval=100, \
    widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()

    if which==1:
        """Exponential"""
        size=15
        sizetitle=18
        b=2020
        xticks = np.arange(b, 31+b, 15)
        fig, ((ax1, ax4, ax3, ax7),(ax2, ax5, ax6, ax8)) = plt.subplots(2,4, figsize=(15,7))
        plt.subplots_adjust(hspace=0.4, wspace=0.6)
        # werte = fkt(t, args)+fkt(t-1, args)+fkt(t-2, args)+fkt(t-3, args)+fkt(t-4, args)+fkt(t-5, args)+fkt(t-6, args)+fkt(t-7, args)+fkt(t-8, args)+fkt(t-9, args) + fkt(t-10, args) + fkt(t-11, args) + fkt(t-12, args)
        # werte2 = fkt(t,args)
        # werte3 = np.zeros_like(t)
        # for i in np.arange(len(t)):
        #     if np.floor(t[i])>np.floor(t[i-1]) and t[i] < 13:
        #         for j in np.arange(10):
        #             werte3[i+j] = 1

        v = growth(t, args)
        ax1.set_xticks(xticks)
        ax1.set_title('Growth rate\n'r'$\nu_{exp} (t)$', size=sizetitle)            # Titel fuer Plots
        #fig.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax1.set_xlim(t[0]+b, 30+b)                            # Achsenlimits x-Achse
        ax1.set_ylim(0, 50*10**6)
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax1.set_ylabel('$c_{exp} = 0.428$ (realistic) \n \n 'r'$\nu_{exp}(t)$ in $\frac{1}{year}$', size=size)   # Beschriftung y-Achse
        ax1.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax1.plot(t+b, v)

        args['r'] = 0.17
        args['a_0'] = args['a_00']*0.447/args['r']
        v = growth(t, args)
        ax2.set_xticks(xticks)
        #fig.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax2.set_xlim(t[0]+b, 30+b)                            # Achsenlimits x-Achse
        ax2.set_ylim(0, 50*10**5)
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax2.set_ylabel('$c_{exp} = 0.17$ \n \n 'r'$\nu_{exp}(t)$ in $\frac{1}{year}$', size=size)   # Beschriftung y-Achse
        ax2.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax2.plot(t+b, v)

        args['r'] = 0.447
        args['a_0'] = args['a_00']*0.447/args['r']
        v2 = np.cumsum(growth(t, args)*dx)+136617
        ax4.set_xticks(xticks)
        ax4.set_title('Total amount of EVs\n'r'$N^{tot}_{exp}(t)$', size=sizetitle)            # Titel fuer Plots
        #fg.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax4.set_xlim(t[0]+b, 30+b)                            # Achsenlimits x-Achse
        ax4.set_ylim(0, 80*10**6)
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax4.set_ylabel(r'$N_{exp}^{tot}(t)$', size=size)   # Beschriftung y-Achse
        ax4.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax4.hlines(47.7*10**6,2000, 2033.1, 'r', ls='--')
        ax4.vlines(2033.1, 0, 47.7*10**6, 'r', ls='--')
        ax4.plot(t+b, v2)

        #f = fkt(t,args)
        v = growth(t,args)
        g = faltung(growth,NSEF,t,args)
        ax3.set_title('Net emission rate\n$g_{exp}(t)$', size = sizetitle)            # Titel fuer Plots
        ax3.set_xlim(t[0]+2020, 2050)                            # Achsenlimits x-Achse
        ax3.set_ylim(0, 10**8)
        ax3.set_xticks(xticks)

        #round_sig(np.max(g)/1000*1.2))
                                    # Achsenlimits y-Achse
        ax3.set_ylabel(r'$g_{exp}(t)$ in $\frac{tons\ CO_2}{year}$', size = size)   # Beschriftung y-Achse
        ax3.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax3.plot(t+2020, g/1000)

        G = dx*np.cumsum(g)
        ax7.set_title('Net total emissions\n$G_{exp}(t)$', size=sizetitle)            # Titel fuer Plots
        ax7.set_xlim(t[0]+2020, 2050)                            # Achsenlimits x-Achse
        ax7.set_ylim(-10, 10**9)                             # Achsenlimits y-Achse
        ax7.set_ylabel(r'$G_{exp}(t)$ in $tons\ CO_2$', size = size)   # Beschriftung y-Achse
        ax7.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax7.plot(t+2020, G/1000)
        ax7.set_xticks(xticks)


        args['r'] = 0.17
        args['a_0'] = args['a_00']*0.447/args['r']
        args['a_0']

        V2 = np.cumsum(growth(t, args)*dx)+136617
        ax5.set_xticks(xticks)
        #ax5.set_title('Total amount of EVs with $c_{exp} = 0.17$', size=sizetitle)            # Titel fuer Plots
        #fig.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax5.set_xlim(t[0]+b, 30+b)                            # Achsenlimits x-Achse
        ax5.set_ylim(0, 80*10**6)
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        ax5.set_ylabel('$N^{tot}_{exp}(t)$', size=size)   # Beschriftung y-Achse
        ax5.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax5.plot(t+b, V2)
        ax5.hlines(47.7*10**6,t0, b + 28.75732428804782, 'r', ls='--')
        ax5.vlines(b + 28.75732428804782, 0, 47.7*10**6, 'r', ls='--')

        #f = fkt(t,args)
        v = growth(t,args)
        g = faltung(growth,NSEF,t,args)
        ax6.set_yticks(np.linspace(-1*10**6,1*10**6,11))

        ax6.set_ylim(- 5*10**5, 5*10**5)
        ax6.set_xlim(t[0]+2020, 2050)                            # Achsenlimits x-Achse
        ax6.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        ax6.set_xticks(xticks)

        #round_sig(np.max(g)/1000*1.2))
                                    # Achsenlimits y-Achse
        ax6.set_ylabel(r'$g_{exp}(t)$ in $\frac{tons\ CO_2}{year}$', size = size)   # Beschriftung y-Achse
        ax6.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax6.plot(t+2020, g/1000)
        ax6.axhline(y=0, c='k', ls='--')

        G = dx*np.cumsum(g)
        ax8.set_xlim(t[0]+2020, 2050)                            # Achsenlimits x-Achse
        ax8.set_ylim(-5*10**6, 5*10**6)                             # Achsenlimits y-Achse
        ax8.set_ylabel(r'$G_{exp}(t)$ in $tons\ CO_2$', size = size)   # Beschriftung y-Achse
        ax8.set_xlabel(r'Time $t$ in years', size = size)                # Beschriftung x-Achse
        ax8.plot(t+2020, G/1000)
        ax8.set_xticks(xticks)
        ax8.axhline(y=0, c='k', ls='--')
        ax8.annotate(r"$\Delta t_{a} = 14.2$ years", xy=(2034.2, 0), xytext=(2022.4, 3*10**6),arrowprops=dict(arrowstyle="->"), size = 15)
        ax1.text(0.07, 0.93, 'a)', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,size =13)
        ax2.text(0.07, 0.93, 'e)', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes,size =13)
        ax3.text(0.07, 0.93, 'c)', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes,size =13)
        ax4.text(0.07, 0.93, 'b)', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes,size =13)
        ax5.text(0.07, 0.93, 'f)', horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,size =13)
        ax6.text(0.07, 0.93, 'g)', horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes,size =13)
        ax7.text(0.07, 0.93, 'd)', horizontalalignment='center', verticalalignment='center', transform=ax7.transAxes,size =13)
        ax8.text(0.07, 0.93, 'h)', horizontalalignment='center', verticalalignment='center', transform=ax8.transAxes,size =13)
        labelsize = 15
        ax2.tick_params(axis='both', which='major', labelsize=labelsize)
        ax3.tick_params(axis='both', which='major', labelsize=labelsize)
        ax4.tick_params(axis='both', which='major', labelsize=labelsize)
        ax5.tick_params(axis='both', which='major', labelsize=labelsize)
        ax6.tick_params(axis='both', which='major', labelsize=labelsize)
        ax1.tick_params(axis='both', which='major', labelsize=labelsize)
        ax7.tick_params(axis='both', which='major', labelsize=labelsize)
        ax8.tick_params(axis='both', which='major', labelsize=labelsize)
        plt.savefig("/Users/diez/X-Uni/2020_0SoSe/BA/Finalfigs/04expgrowth.pdf")


    elif which==2:
        size=10
        sizetitle=13
        b=2020
        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(1, 1, 1)
        data = []
        f = fkt(t,args)
        for i in np.arange(100):
            args['r'] = i/500
            args['a_0'] = args['a_00']*0.447/args['r']
            v = growth(t,args)
            g = faltung(v,f,t)
            G = dx*np.cumsum(g)
            data.append([args['r'], amor_time(t,G)])
            bar.update(i*0.9)
        for i in np.arange(10):
            args['r'] = 0.2+0.03*i
            args['a_0'] = args['a_00']*0.447/args['r']
            v = growth(t,args)
            g = faltung(v,f,t)
            G = dx*np.cumsum(g)
            data.append([args['r'], amor_time(t,G)])
            bar.update(i+90)



        data = np.array(data)
        data = data[data[:,1]>0]
        fig.suptitle('Amortization time in dependency of r')
        ax.set_xlim(0,0.5)                            # Achsenlimits x-Achse
        ax.set_ylim(2020, 2020+np.max(data[:,1]))                             # Achsenlimits y-Achse
        ax.set_ylabel(r'Amortization time in years', size=size)   # Beschriftung y-Achse
        ax.set_xlabel(r'Rate r',size=20)                # Beschriftung x-Achse
        ax.plot(data[:,0],2020+data[:,1])
        ax.vlines(0.447, 0, np.max(data[:,1]), colors='r', ls = '--')
        ax.vlines(0.17, 0+b, 20.5 +b, colors='r', ls = '--')
        plt.show()
        bar.finish()
    elif which==3:
        size=10
        sizetitle=13
        b=2020
        fig = plt.figure(figsize=(15,8))
        ax = fig.add_subplot(1, 1, 1)
        data = []
        f = fkt(t,args)
        for i in np.arange(100):
            args['r'] = i/500
            args['a_0'] = args['a_00']*0.447/args['r']
            v = growth(t,args)
            g = faltung(v,f,t)
            G = dx*np.cumsum(g)
            data.append([1/args['r']*np.log(47.7*10**6/(args['a_0'])), amor_time(t,G)])
            bar.update(i*0.9)
        for i in np.arange(10):
            args['r'] = 0.2+0.03*i
            args['a_0'] = args['a_00']*0.447/args['r']
            v = growth(t,args)
            g = faltung(v,f,t)
            G = dx*np.cumsum(g)
            data.append([args['r'], amor_time(t,G)])
            bar.update(i+90)
        data = np.array(data)
        data = data[data[:,1]>0]
        ax.vlines(2020+1/0.447*np.log(47.7*10**6/(args['a_00']*0.477/0.477)), 0, np.max(data[:,1]), colors='r', ls='--')
        ax.vlines(2020+1/0.17*np.log(47.7*10**6/(args['a_00']*0.447/0.17)), 0, 20.5, colors='r',ls='--')

        fig.suptitle('Amortization time in dependency of year of transition')
        ax.set_xlim(2020,2100)                            # Achsenlimits x-Achse
        ax.set_ylim(2020, 2020+np.max(data[:,1]))                             # Achsenlimits y-Achse
        ax.set_ylabel(r'year of $CO_2$-Amortization', size=size)   # Beschriftung y-Achse
        ax.set_xlabel(r'year of transition',size=size)                # Beschriftung x-Achse
        ax.plot(data[:,0]+2020,2020+data[:,1])

        plt.show()
        bar.finish()

    elif which==4:
        N=3 #100
        M=2 #10
        size=10
        sizetitle=13
        b=2020
        fig, ((ax1),(ax2)) = plt.subplots(2,1, figsize=(15,9))
        plt.subplots_adjust(hspace=0.4, wspace=0.4)
        data = []
        #f = fkt(t,args)
        for i in np.arange(N):
            args['r'] = i/500
            #args['a_0'] = args['a_00']*0.447/args['r']
            v = growth(t,args)
            g = faltung(v,NSEF,t,args)
            G = dx*np.cumsum(g)
            data.append([args['r'], amor_time(t,G)])
            bar.update(i*0.9)
        for i in np.arange(M):
            args['r'] = 0.2+0.03*i
            #args['a_0'] = args['a_00']*0.447/args['r']
            v = growth(t,args)
            g = faltung(v,NSEF,t,args)
            G = dx*np.cumsum(g)
            data.append([args['r'], amor_time(t,G)])
            bar.update(i+90)
        data = np.array(data)
        data = data[data[:,1]>0]
        ax1.set_xlim(0,0.5)
        ax1.set_ylabel(r'year of $CO_2$-Amortization', size=size)   # Beschriftung y-Achse
                           # Achsenlimits x-Achse
        ax1.set_ylim(2020, 2020+np.max(data[:,1]))                             # Achsenlimits y-Achse
        ax1.set_title(r'Amortization time in dependency of $c_{exp}$', size=size)   # Beschriftung y-Achse
        ax1.set_xlabel('growth factor $c_{exp}$',size=size)                # Beschriftung x-Achse
        ax1.plot(data[:,0],2020+data[:,1])
        ax1.vlines(0.447, 0+b, b+np.max(data[:,1]), colors='r', ls = '--')
        ax1.vlines(0.17, 0+b, 20.5 +b, colors='r', ls = '--')
        ax1.text(0.447-0.01, b+10, '$c_{exp} = 0.447$', ha='center', va='center',rotation='vertical')
        ax1.text(0.17-0.01, b+10, '$c_{exp} = 0.17$', ha='center', va='center',rotation='vertical')
        bar.start()
        data = []
        #f = fkt(t,args)
        for i in np.arange(100):
            args['r'] = i/500
            #args['a_0'] = args['a_00']*0.447/args['r']
            v = growth(t,args)
            g = faltung(v,f,t)
            G = dx*np.cumsum(g)
            data.append([1/args['r']*np.log(47.7*10**6/(args['a_0'])), amor_time(t,G)])
            bar.update(i*0.9)
        for i in np.arange(10):
            args['r'] = 0.2+0.03*i
            #args['a_0'] = args['a_00']*0.447/args['r']
            v = growth(t,args)
            g = faltung(v,f,t)
            G = dx*np.cumsum(g)
            data.append([args['r'], amor_time(t,G)])
            bar.update(i+90)
        data = np.array(data)
        data = data[data[:,1]>0]
        ax2.vlines(2020+1/0.447*np.log(47.7*10**6/(args['a_00']*0.477/0.477)), 0, np.max(data[:,1]), colors='r', ls='--')
        ax2.vlines(2020+1/0.17*np.log(47.7*10**6/(args['a_00']*0.447/0.17)), 0, 20.5, colors='r',ls='--')

        ax2.set_title('Amortization time in dependency of year of transition', size = size)
        ax2.set_xlim(2020,2100)                            # Achsenlimits x-Achse
        ax2.set_ylim(2020, 2020+np.max(data[:,1]))                             # Achsenlimits y-Achse
        ax2.set_ylabel(r'year of $CO_2$-Amortization', size=size)   # Beschriftung y-Achse
        ax2.set_xlabel(r'year of transition',size=size)                # Beschriftung x-Achse
        ax2.plot(data[:,0]+2020,2020+data[:,1])
        ax2.vlines(2020+1/0.447*np.log(47.7*10**6/(args['a_00']*0.477/0.477)), b, b+np.max(data[:,1]), colors='r', ls='--')
        ax2.text(2019+1/0.447*np.log(47.7*10**6/(args['a_00']*0.477/0.477)), b+10, '$c_{exp} = 0.447$', ha='center', va='center',rotation='vertical')
        ax2.vlines(2020+1/0.17*np.log(47.7*10**6/(args['a_00']*0.447/0.17)), b+0, b+20.5, colors='r',ls='--')
        ax2.text(2019+1/0.17*np.log(47.7*10**6/(args['a_00']*0.477/0.17)), b+10, '$c_{exp} = 0.17$', ha='center', va='center',rotation='vertical')
        plt.savefig("/Users/diez/X-Uni/2020_0SoSe/BA/Finalfigs/05exprate2.pdf")
        bar.finish()

        print('Plot 1 fertig')
        bar.finish()


    elif which ==5:
        size=15
        sizetitle=20
        b=2020
        r = np.linspace(10**(-2),0.5,1000)
        fig, ((ax1),(ax2)) = plt.subplots(2,1, figsize=(15,10))
        plt.subplots_adjust(hspace=0.45, wspace=0.4)
        ax1.set_xlim(0,0.5)
        ax1.set_ylabel(r'year of finished transition', size=size)   # Beschriftung y-Achse
                           # Achsenlimits x-Achse
        ax1.set_ylim(b, b+80)                             # Achsenlimits y-Achse
        ax1.set_title(r'year of finished transition', size=sizetitle)   # Beschriftung y-Achse
        ax1.set_xlabel('growth factor $c_{exp}$',size=size)                # Beschriftung x-Achse
        ax1.plot(r, b+1/r*np.log(47.7*10**6/(args['a_00'])))
        ax1.vlines(0.447, 0+b, b+1/0.447*np.log(47.7*10**6/(args['a_00']*0.447/0.447)), colors='r', ls = '--')
        ax1.vlines(0.17, 0+b, b+1/0.17*np.log(47.7*10**6/(args['a_00'])) , colors='r', ls = '--')
        ax1.text(0.447, b+37, '$c_{exp} = 0.428$', ha='center', va='center',rotation='vertical',size=size)
        ax1.text(0.17, b+55, '$c_{exp} = 0.17$', ha='center', va='center',rotation='vertical',size=size)
        #
        ##################
        #
        M=33
        N=6
        O =49
        P=27
        bar.start()
        data = []
        #f = fkt(t,args)
        for i in np.arange(M): #0.3254826
            args['r'] =0.01*i
            #args['a_0'] = args['a_00']*0.447/args['r']
            v = growth(t,args)
            g = faltung(growth,NSEF,t,args)
            G = dx*np.cumsum(g)
            data.append([1/args['r']*np.log(47.7*10**6/(args['a_0'])), amor_time(t,G)])
            data
            bar.update(i/(M+N+O+P)*100)
        for i in np.arange(N):
            args['r'] = 0.32+0.001*i
            #args['a_0'] = args['a_00']*0.447/args['r']
            g = faltung(growth,NSEF,t,args)
            G = dx*np.cumsum(g)
            data.append([1/args['r']*np.log(47.7*10**6/(args['a_0'])), amor_time(t,G)])
            data
            bar.update((i+M)/(M+N+O+P)*100)
        for i in np.arange(O):

            args['r'] = 0.325+0.00001*i
            #args['a_0'] = args['a_00']*0.447/args['r']
            g = faltung(growth,NSEF,t,args)
            G = dx*np.cumsum(g)
            data.append([1/args['r']*np.log(47.7*10**6/(args['a_0'])), amor_time(t,G)])
            data
            bar.update((i+M+N)/(M+N+O+P)*100)
        for i in np.arange(P):
            args['r'] = 0.32548+0.0000001*i
            #args['a_0'] = args['a_00']*0.447/args['r']
            g = faltung(growth,NSEF,t,args)
            G = dx*np.cumsum(g)
            data.append([1/args['r']*np.log(47.7*10**6/(args['a_0'])), amor_time(t,G)])
            data
            bar.update((i+M+N+O)/(M+N+O+P)*100)

        data = np.array(data)
        #data = data[data[:,1]>0]
        ax2.vlines(2020+1/0.447*np.log(47.7*10**6/(args['a_00']*0.477/0.477)), 0, np.max(data[:,1]), colors='r', ls='--')
        ax2.vlines(2020+1/0.17*np.log(47.7*10**6/(args['a_00'])), 0, 15.1, colors='r',ls='--')
        ax2.set_title('Amortization time in dependency of year of finished transition', size = sizetitle)
        ax2.set_xlim(2020,2100)                            # Achsenlimits x-Achse
        ax2.set_ylim(2020, 2080)                             # Achsenlimits y-Achse
        ax1.tick_params(axis='both', which='major', labelsize=13)
        ax2.tick_params(axis='both', which='major', labelsize=13)
        ax1.set_yticks(np.arange(2020,2081,20))
        ax2.set_yticks(np.arange(2020,2081,20))
        ax2.set_ylabel(r'year of $CO_2$-Amortization', size=size)   # Beschriftung y-Achse
        ax2.set_xlabel(r'year of finished transition',size=size)                # Beschriftung x-Achse
        ax2.plot(data[:,0]+2020,2020+data[:,1])
        ax2.vlines(2020+1/0.447*np.log(47.7*10**6/(args['a_00']*0.477/0.477)), b, b+np.max(data[:,1]), colors='r', ls='--')
        ax2.text(2019+1/0.447*np.log(47.7*10**6/(args['a_00']*0.477/0.477)), b+25, '$c_{exp} = 0.428$', ha='center', va='center',rotation='vertical',size=size)
        ax2.vlines(2020+1/0.17*np.log(47.7*10**6/(args['a_00'])), b+0, b+14.2, colors='r',ls='--')
        ax2.text(2020+1/0.17*np.log(47.7*10**6/(args['a_00'])), b+30, '$c_{exp} = 0.17$', ha='center', va='center',rotation='vertical',size=size)
        ax1.text(0.02, 0.93, 'a)', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,size =15)
        ax2.text(0.02, 0.93, 'b)', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes,size =15)
        plt.savefig("/Users/diez/X-Uni/2020_0SoSe/BA/Finalfigs/05exprate.pdf")
        bar.finish()
    plt.show()

if __name__ == "__main__":
    main()
