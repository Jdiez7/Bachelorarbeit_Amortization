import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons
from math import floor, log10

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
    #lel = 0.401*(neutral-(t+year))/(neutral-2019)

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
    #
    # j = np.searchsorted(t, lifetime+breite*dx, side='left')
    # y[j] += recycling/dx - breite*recycling
    # for m in np.arange(int(breite)):
    #     y[j+m] += recycling


    return y


"""ab hier könnte auch alles importiert werden, weiß gerade nur nicht wie das geht"""
def faltung2(v, NSEF, t, args):
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
        f = NSEF(t,t[i],args)
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
    r, a_0 = args['r'], args['a_0']
    a = np.zeros(len(t))
    i = np.searchsorted(t,0,side='left')
    a[i:] = a_0*np.exp(r*t[i:])
    return a

def growth_lin(t, args):
    """Wachstumsfunktion Zubau; hier Annahme linear
    t           Zeitenarray
    c           Zubaurate"""
    c = args['c']
    a = np.zeros(len(t))
    a+=c
    i = np.searchsorted(t,0,side='left')
    a[:i]=0
    # for i in np.arange(len(t)):
    #     if t[i]>0:
    #        a[i:] = c
    #        break
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
    args['delta'] = 0                 # breite der Deltafunktion
    args['n'] = 10000

    # REALISTISCH
    args['lifekm'] = 230000             # Lebensdauer in km
    args['driving'] = 14000             # gefahrene Kilometer pro Jahr
    args['breite'] = 0 #args['n']/400
    args['neutral'] = 2050
    args['neu_ausstoß'] = 14700          # kgCO_2-Ausstoß EV
    args['neu_ausstoß_ICE'] = 9100      # kgCO_2-Ausstoß ICE
    args['verbrauch'] = 0.165            # kWh pro km
    args['ausstoss_ICE'] = 0.1306          # kg pro km
    args['recycling'] = 0             # kg CO_2 für recycling

    # logistisches Wachstum
    args['k'] = 8.79255838*10**(-9)                     # r-Wert für logistisches Wachstum
    args['amax'] = 47.7*10**6               # Endwert für log Wachstum
    args['log_0'] = 688.317359                         # Anfangszahl von e-Autos bei exp. growth


    # Aus vorherigem Dokument
    args['a_0'] = 1                     # Anfangszahl von e-Autos bei exp. growth
    args['r'] = 0.4                     # growth rate
    args['c'] = 1                       # Zubaurate Units pro Zeit - lin
    # Grundgerüst
    global t0
    t0 = -2

    t = np.linspace(t0,50,args['n'])    # Zeitenarray
    dx = (t[-1]-t[0])/(len(t)-1)

    scenario = 1
    which = int(input('1=single, 2= Int, 3=growth, 4=Faltung, 5=total emmissions, 6= r-abh., 7 =k-abh, 8=lifetime abhgk., 9 = neuaussto abhgk., 10=neutral jahr'))
     # 1=single, 2= Int, 3=growth, 4=Faltung, 5=total emmissions, 6= r-abh., 7 =k-abh, 8=lifetime abhgk., 9 = neuaussto abhgk., 10=neutral jahr,

    fkt = single_simple_real   #_delta, _block
    growth = growth_lin         #_lin, _exp, log
    NSEF = fkt

    # Erstellen der Plotumgebung
    # Nutzen von einem Subplots

    # # RadioButton
    # axcolor = 'lightgoldenrodyellow'
    # rax = plt.axes([0.05, 0.7, 0.15, 0.15], facecolor=axcolor)
    # radio = RadioButtons(rax, ('1', '2','3','4','5','6','7','8','9','10'))
    # radio.on_clicked(func)

    # Achsenbereiche festlegen und Beschriftungen setzen
    if which==0:
        t=np.linspace(-100,15,10000)
        v = growth(t, args)
        plt.plot(t,v)
        g = faltung(growth,NSEF,t,args)
        plt.plot(t,g)
        plt.plot(t,NSEF(t,0,args))

        fig, (ax) = plt.subplots(1, 1, figsize=(12,7.5))
        plt.plot(t,np.cumsum(g))
        amor_time(t,np.cumsum(g))
    if which==1:
        size=13
        sizetitle=15
        b=2020
        xticks = np.arange(b, 16+b, 5)
        fig, ((ax01, ax02, ax03),(ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(3, 3, figsize=(12,8))
        werte = fkt(t,0, args)+fkt(t-1,1, args)+fkt(t-2,2, args)+fkt(t-3,3, args)+fkt(t-4,4, args)+fkt(t-5,5, args)+fkt(t-6,6, args)+fkt(t-7,7, args)+fkt(t-8,8, args)+fkt(t-9,9, args) + fkt(t-10,10, args) + fkt(t-11,11, args) + fkt(t-12,12, args)
        werte2 = fkt(t,0,args)
        werte3 = np.zeros_like(t)
        for i in np.arange(len(t)):
            if np.floor(t[i])>np.floor(t[i-1]) and t[i] < 14 and t[i]>-0.1:
                for j in np.arange(10):
                    werte3[i+j] = 1




        ax01.set_title('Discrete growth \n \n', size=sizetitle)
        ax01.plot(t+b,werte3*3)
        ax01.set_xlim(t[0]+b, 16+b)
        ax01.set_ylim(0,2)
        ax01.set_yticks(np.arange(0,3,1))
        ax01.set_xticks(xticks)
        ax01.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax01.set_ylabel(r'Growth rate ''\n'r'$\nu (t)$ in $\frac{EVs}{Year}$', size=size)                # Beschriftung x-Achse


                   # Beschriftung x-Achse
        v = growth(t, args)
        # fig.suptitle('Exponentielles Wachstum mit Wachstumsrate {0}'.format(args['r']))            # Titel fuer Plots
        ax02.set_title('Continuous growth \n 'r'$c_\mathrm{lin} = \frac{1}{Year}$ ''\n',  size=sizetitle )            # Titel fuer Plots
        #fig.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax02.set_xlim(t[0]+b, 16+b)                            # Achsenlimits x-Achse
        ax02.set_ylim(0, round_sig(np.max(v)*2))
        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        #ax02.set_ylabel(r'Construction of EVs in years$^{-1}$', size=size)   # Beschriftung y-Achse
        ax02.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax02.set_yticks(np.arange(0,3,1))

        ax02.plot(t+b, v)
        ax02.set_xticks(xticks)


        args['c']=2*10**6
        v = growth(t, args)
        # fig.suptitle('Exponentielles Wachstum mit Wachstumsrate {0}'.format(args['r']))            # Titel fuer Plots
        ax03.set_title('Continuous growth \n'r' $c_\mathrm{lin} = \frac{2 Mio.}{Year}$''\n', size=sizetitle )            # Titel fuer Plots
        #fig.suptitle('lineares Wachstum mit Zubaurate {0}'.format(args['c']))
        ax03.set_xlim(t[0]+b, 16+b)                            # Achsenlimits x-Achse
        ax03.set_ylim(0, round_sig(np.max(v)*2))
        ax03.set_yticks(np.arange(0,3*args['c'],1*args['c']))

        #ax.set_ylim(0, np.max(v)*4)                    # Achsenlimits y-Achse
        #ax03.set_ylabel(r'Construction of EVs in years$^{-1}$', size=size)   # Beschriftung y-Achse
        ax03.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax03.plot(t+b, v)
        ax03.set_xticks(xticks)




        ax1.set_xlim(t[0]+b, 16+b)                            # Achsenlimits x-Achse
        ax1.set_ylim(-20, 10)                             # Achsenlimits y-Achse
        ax1.set_xticks(xticks)
        ax1.set_yticks(np.arange(-20, 10, 8))
        ax1.set_ylabel(r'Net $CO_2$ emission rate ''\ng(t)'r' in $\frac{tons\ CO_2}{Year}$', size=size)   # Beschriftung y-Achse
        ax1.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax1.plot(t+b, werte/1000, linewidth = 2)
        ax1.axhline(y=0, color='k', ls='--')

        args['c']=1
        #f = fkt(t,0,args)
        v = growth(t,args)
        g = faltung(growth,NSEF,t,args)
        ax2.set_xticks(xticks)
        ax2.set_yticks(np.arange(-16, 10, 8))
        ax2.set_xlim(t[0]+b, 16+b)                            # Achsenlimits x-Achse
        ax2.set_ylim(-20, 10)
        #ax2.set_ylabel(r'Net $CO_2$-emission in $\frac{t}{Year}$', size=size)   # Beschriftung y-Achse
        ax2.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax2.plot(t+b, g/1000)
        ax2.axhline(y=0, c='k', ls='--')

        args['c']=2*10**6
        #f = fkt(t,args)
        v = growth(t,args)
        g = faltung(growth,NSEF,t,args)
        ax3.set_xticks(xticks)
        ax3.set_yticks(np.arange(-16*args['c'], 10*args['c'], 8*args['c']))
        ax3.set_xlim(t[0]+b, 16+b)                            # Achsenlimits x-Achse
        ax3.set_ylim(-20*args['c'], 10*args['c'])
        #ax3.set_ylabel(r'Net $CO_2$-emission in $\frac{t}{Year}$', size=size)   # Beschriftung y-Achse
        ax3.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax3.plot(t+b, g/1000)
        ax3.axhline(y=0, c='k', ls='--')





        ax4.set_xlim(t[0]+b, 16+b)                            # Achsenlimits x-Achse
        ax4.set_ylim(-10, 20)                             # Achsenlimits y-Achse
        ax4.set_xticks(xticks)
        ax4.set_yticks(np.arange(-8, 20, 8))
        ax4.set_ylabel(r'Total $CO_2$ emissions ''\nG(t) in tons $CO_2$', size=size)   # Beschriftung y-Achse
        ax4.set_xlabel(r'Time $t$ in years',  size=size)                # Beschriftung x-Achse
        ax4.plot(t+b, dx*np.cumsum(werte)/1000, linewidth = 2)
        ax4.axhline(y=0, color='k', ls='--')

        args['c']=1
        #f = fkt(t,args)
        #F = dx*np.cumsum(fkt(t,0, args))
        v = growth(t,args)
        g = faltung(growth,NSEF,t,args)
        G = dx*np.cumsum(g)
        ax5.set_xlim(t[0]+b, 16+b)                            # Achsenlimits x-Achse
        ax5.set_ylim(-10, 20)                             # Achsenlimits y-Achse
        ax5.set_xticks(xticks)
        ax5.set_yticks(np.arange(-8, 20, 8))
        #ax5.set_ylabel(r'Total $CO_2$ emission in t', size=size)   # Beschriftung y-Achse
        ax5.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax5.plot(t+b, G/1000)
        ax5.axhline(y=0, c='k', ls='--')

        args['c']=2*10**6
        #f = fkt(t,args)
        #F = dx*np.cumsum(fkt(t, args))
        v = growth(t,args)
        g = faltung(growth,NSEF,t,args)
        G = dx*np.cumsum(g)
        ax6.set_xlim(t[0]+b, 16+b)                            # Achsenlimits x-Achse
        ax6.set_ylim(-10*args['c'], 20*args['c'])                             # Achsenlimits y-Achse
        ax6.set_xticks(xticks)
        ax6.set_yticks(np.arange(-8*args['c'], 20*args['c'], 8*args['c']))
        #ax6.set_ylabel(r'Total $CO_2$ emission in t', size=size)   # Beschriftung y-Achse
        ax6.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax6.plot(t+b, G/1000)
        ax6.axhline(y=0, c='k', ls='--')
        plt.subplots_adjust(hspace=0.45, wspace=0.25)
        ax5.annotate(r"$\Delta t_{a} = 9.9$ years", xy=(2029.9, 0), xytext=(2029, 10),arrowprops=dict(arrowstyle="->"), size = 8)
        ax6.annotate(r"$\Delta t_{a} = 9.9$ years", xy=(2029.9, 0), xytext=(2029, 10*2*10**6),arrowprops=dict(arrowstyle="->"), size = 8)
        ax01.text(0.05, 0.93, 'a)', horizontalalignment='center', verticalalignment='center', transform=ax01.transAxes,size =13)
        ax02.text(0.05, 0.93, 'b)', horizontalalignment='center', verticalalignment='center', transform=ax02.transAxes,size =13)
        ax03.text(0.05, 0.93, 'c)', horizontalalignment='center', verticalalignment='center', transform=ax03.transAxes,size =13)
        ax1.text(0.05, 0.93, 'd)', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,size =13)
        ax2.text(0.05, 0.93, 'e)', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes,size =13)
        ax3.text(0.05, 0.93, 'f)', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes,size =13)
        ax4.text(0.05, 0.93, 'g)', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes,size =13)
        ax5.text(0.05, 0.93, 'h)', horizontalalignment='center', verticalalignment='center', transform=ax5.transAxes,size =13)
        ax6.text(0.05, 0.93, 'i)', horizontalalignment='center', verticalalignment='center', transform=ax6.transAxes,size =13)
        ax1.tick_params(axis='both', which='major', labelsize=13)
        ax2.tick_params(axis='both', which='major', labelsize=13)
        ax3.tick_params(axis='both', which='major', labelsize=13)
        ax4.tick_params(axis='both', which='major', labelsize=13)
        ax5.tick_params(axis='both', which='major', labelsize=13)
        ax6.tick_params(axis='both', which='major', labelsize=13)
        ax01.tick_params(axis='both', which='major', labelsize=13)
        ax02.tick_params(axis='both', which='major', labelsize=13)
        ax03.tick_params(axis='both', which='major', labelsize=13)
        plt.savefig("/Users/diez/X-Uni/2020_0SoSe/BA/Finalfigs/02lingrowth.pdf")


    if which==2:
        size=13
        sizetitle=15
        b=2020
        xticks = np.arange(b, 21+b, 5)
        fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2,2, figsize = (10,6))
        plt.subplots_adjust(hspace=0.4, wspace=0.25)
        werte = fkt(t,0, args)                    # Titel fuer Plots
        werte2 = np.cumsum(dx*werte)

        ax1.set_title('Single vehicle \n', size = sizetitle)
        ax1.set_xticks(xticks)
        ax1.set_yticks(np.arange(-20, 10, 2))
        ax1.set_xlim(t[0]+b, 20+b)                            # Achsenlimits x-Achse
        ax1.set_ylim(round_sig(1.4*np.min(werte)/1000), round_sig(1.2*(args['neu_ausstoß']-args['neu_ausstoß_ICE'])/1000))                             # Achsenlimits y-Achse
        ax1.set_ylabel(r'Net emission rate''\ng(t) 'r'in $\frac{tons\ CO_2}{Year}$', size=size)   # Beschriftung y-Achse
        ax1.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax1.plot(t+b, werte/1000)
        ax1.axhline(y=0, c='k', ls='--')


        args['c']=1
        #f = fkt(t,args)
        v = growth(t,args)
        g = faltung(growth,NSEF,t,args)

        ax2.set_title('Linear growth\n $c_{lin} = 1$', size= sizetitle)
        ax2.set_xticks(xticks)
        ax2.set_yticks(np.arange(-16, 10, 8))

        ax2.set_ylim(round_sig(-3*np.max(g)/1000), 8)                             # Achsenlimits y-Achse
        ax2.set_xlim(t[0]+b, 20+b)                            # Achsenlimits x-Achse
        #ax2.set_ylabel(r'Net $CO_2$-emission in $\frac{t}{Year}$', size=size)   # Beschriftung y-Achse
        ax2.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax2.plot(t+b, g/1000)
        ax2.axhline(y=0, c='k', ls='--')


        ax3.set_xticks(xticks)
        ax3.set_yticks(np.arange(-16, 10, 8))
        ax3.set_xlim(t[0]+b, 20+b)                            # Achsenlimits x-Achse
        ax3.set_ylim(round_sig(-3*np.max(werte2)/1000), 8)                             # Achsenlimits y-Achse
        ax3.set_ylabel('Net total emissions\nG(t) in tons $CO_2$', size=size)   # Beschriftung y-Achse
        ax3.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax3.plot(t+b, werte2/1000)
        ax3.axhline(y=0, c='k', ls='--')

        #f = fkt(t,args)
        #F = dx*np.cumsum(fkt(t, args))
        v = growth(t,args)
        g = faltung(growth,NSEF,t,args)
        G = dx*np.cumsum(g)
        ax4.set_xlim(t[0]+b, 16+b)                            # Achsenlimits x-Achse
        ax4.set_ylim(-10, 20)                             # Achsenlimits y-Achse
        ax4.set_xticks(xticks)
        ax4.set_yticks(np.arange(-8, 20, 8))
        #ax4.set_ylabel(r'Total $CO_2$ emission in t', size=size)   # Beschriftung y-Achse
        ax4.set_xlabel(r'Time $t$ in years', size=size)                # Beschriftung x-Achse
        ax4.plot(t+b, G/1000)
        ax4.axhline(y=0, c='k', ls='--')
        ax4.annotate(r"$\Delta t_{a} = 9.9$ years", xy=(2029.9, 0), xytext=(2031, 6),arrowprops=dict(arrowstyle="->"), size = 10)
        ax3.annotate(r"$\Delta t_{a} = 5.5$ years", xy=(2025.5, 0), xytext=(2025, 3),arrowprops=dict(arrowstyle="->"), size = 10)
        ax1.text(0.05, 0.93, 'a)', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,size =13)
        ax2.text(0.05, 0.93, 'b)', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes,size =13)
        ax3.text(0.05, 0.93, 'c)', horizontalalignment='center', verticalalignment='center', transform=ax3.transAxes,size =13)
        ax4.text(0.05, 0.93, 'd)', horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes,size =13)
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax3.tick_params(axis='both', which='major', labelsize=12)
        ax4.tick_params(axis='both', which='major', labelsize=12)
        plt.savefig("/Users/diez/X-Uni/2020_0SoSe/BA/Finalfigs/03lingrowth2.pdf")

        #ax4.arrow(2035, 6, -4.39, -6)

    plt.show()

if __name__ == "__main__":
    main()
1
