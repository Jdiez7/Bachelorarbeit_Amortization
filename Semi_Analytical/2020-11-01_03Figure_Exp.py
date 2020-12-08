import numpy as np                          # Modul zur allgemeinen Berechnung.
from matplotlib import pyplot as plt           # Graphikbefehle.def main():
from scipy import optimize

def test_func2(x, a, b):
    return a*x+b

def test_func(x, r):
    """Wachstumsfunktion Zubau; hier Annahme logistisch
    t           Zeitenarray
    r           Exp. Rate"""
    return 136617*np.exp(r*(x-2020))
def test_func3(x, r,x0):
    """Wachstumsfunktion Zubau; hier Annahme logistisch
    t           Zeitenarray
    r           Exp. Rate"""
    return x0*np.exp(r*(x-2020))




def main():


#    x = np.array([2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019 ,2020])
    x = np.array([2011,2012,2013,2014,2015,2016,2017,2018,2019,2020])
    y = np.array([2307, 4541, 7114, 12156, 18948, 25502, 34022, 53861, 83175, 136617])
    x2=np.linspace(2000,2050, 1000)
    size = 18
    headsize = 23
    params, params_covariance = optimize.curve_fit(test_func2, x, np.log(y)) # 29-1000 sind die interessanten Messwerte
    params
    # Erstellen der Plotumgebung
    fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15,5))
    plt.subplots_adjust(hspace= 0.6)
    """Plot 1 - exponentiell """
    ax1.scatter(x, y, 20, label='Data')
    test_func3(x2, params[0], np.exp(params[1]+params[0]*2020))
    ax1.plot(x2, test_func3(x2, params[0], np.exp(params[1]+params[0]*2020)), 'r--',label='Fit')
    #ax.plot(x2, 34022*np.exp(0.439*x2))
    ax1.set_xlabel('Year', size=size)
    ax1.set_ylabel('Total number of EVs', size=size)
    ax1.set_ylim(10**3, 10**6)
    ax1.set_yscale('symlog')
    ax1.set_xlim(2010, 2021)
    #ax.grid(b=None, which='major', axis='both', color='grey', linestyle='-',linewidth=0.5)
    ax1.legend()
    ax1.set_title('Fit for exponential increase',size=headsize)
    """Plot 2 - normal"""
    ax2.scatter(x, y, 20, label='Data')
    ax2.plot(x2, test_func(x2, params[0]), 'r--',label='Fit')
    #ax.plot(x2, 34022*np.exp(0.439*x2))
    ax2.set_xlabel('Year', size=size)
    ax2.set_ylabel('Total number of EVs', size=size)
    ax2.set_ylim(10**3, 10**8)
    ax2.set_xlim(2010, 2035)
    ax2.hlines(4.77*10**7,2010,2035, 'k', ls='--', label='total # of vehicles')
    ax2.set_title('Development until 2035',size=headsize)
    ax2.legend()
    ax1.text(0.05, 0.93, 'a)', horizontalalignment='center', verticalalignment='center', transform=ax1.transAxes,size =13)
    ax2.text(0.05, 0.93, 'b)', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes,size =13)
    ax1.tick_params(axis='both', which='major', labelsize=15)
    ax2.tick_params(axis='both', which='major', labelsize=15)
    plt.savefig("/Users/diez/X-Uni/2020_0SoSe/BA/Finalfigs/011fitexp.pdf")


    plt.show()
    print(params, params_covariance)
    print(np.exp(params[1]+params[0]*2020))

if __name__ == "__main__":
    main()
