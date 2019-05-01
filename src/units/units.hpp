#ifndef UMUQ_UNITS_H
#define UMUQ_UNITS_H

#include "misc/parser.hpp"
#include "lattice.hpp"

namespace umuq
{

/*!
 * \defgroup Units_Module Units module
 *
 * This is the Units Module of UMUQ providing all necessary classes
 * for physical units and their conversion currently supported in UMUQ.
 */

/*!
 * \enum ChargeUnit
 * \ingroup Units_Module
 *
 * \brief The ChargeUnit class
 *
 */
enum class ChargeUnit
{
    /*! The \b Coulomb unit of charge. */
    Coulomb,
    /*! The \b electron unit of charge. */
    electron,
    /*! The \b stat-coulomb unit of charge. */
    statCoulomb
};

/*!
 * \enum  EnergyUnit
 * \ingroup Units_Module
 *
 * \brief The EnergyUnit class
 *
 */
enum class EnergyUnit
{
    /*! The \f$ {\frac{\textbf{amu}~\textbf{A}^2}{\textbf{ps}^2}} \f$ unit of energy. */
    amu_A2_per_ps2,
    /*! The \b erg unit of energy. */
    erg,
    /*! The \b electron-volt unit of energy. */
    eV,
    /*! The \b Hartree unit of energy.*/
    Hartree,
    /*! The \b Joule unit of energy. */
    J,
    /*! The \b kilocalorie per mole unit of energy. */
    kcal_mol
};

/*!
 * \enum LengthUnit
 * \ingroup Units_Module
 *
 * \brief The LengthUnit class
 *
 */
enum class LengthUnit
{
    /*! The \b angstrom unit of length. */
    A,
    /*! The \b Bohr unit of length.*/
    Bohr,
    /*! The \b centimeter unit of length.*/
    cm,
    /*! The \b meter unit of length. */
    m,
    /*! The \b nano-meter unit of length. */
    nm
};

/*!
 * \enum TemperatureUnit
 * \ingroup Units_Module
 *
 * \brief The TemperatureUnit class
 *
 */
enum class TemperatureUnit
{
    /*! The \b Kelvin unit of temperature. */
    K
};

/*!
 * \enum TimeUnit
 * \ingroup Units_Module
 *
 * \brief The TimeUnit class
 *
 */
enum class TimeUnit
{
    /*! The \b femto-second unit of time. */
    fs,
    /*! The \b pico-second unit of time. */
    ps,
    /*! The \b nano-second unit of time. */
    ns,
    /*! The \b second unit of time. */
    s
};

/*!
 * \enum ForceUnit
 * \ingroup Units_Module
 *
 * \brief The ForceUnit class
 *
 */
enum class ForceUnit
{
    /*! The unit of force in \b REAL style */
    Kcal_moleAngstrom,
    /*! The unit of force in \b METAL style */
    eV_Angstrom,
    /*! The unit of force in \b SI style */
    Newtons,
    /*! The unit of force in \b CGS style */
    dynes,
    /*! The unit of force in \b ELECTRON style */
    Hartrees_Bohr,
    /*! The unit of force */
    Rydberg_Bohr
};

/*!
 * \enum UnitStyle
 * \ingroup Units_Module
 *
 * \brief The unit system style
 * It determines the units of all quantities.
 */
enum class UnitStyle
{
    /*!
     * \brief \b REAL style
     *
     * <table>
     * <caption id="multi_row">REAL style</caption>
     * <tr><th> Units             <th> Description
     * <tr><td> mass              <td> grams/mole
     * <tr><td> distance          <td> Angstroms
     * <tr><td> time              <td> femto-seconds
     * <tr><td> energy            <td> Kcal/mole
     * <tr><td> velocity          <td> Angstroms/femto-second
     * <tr><td> force             <td> Kcal/mole-Angstrom
     * <tr><td> torque            <td> Kcal/mole
     * <tr><td> temperature       <td> Kelvin
     * <tr><td> pressure          <td> atmospheres
     * <tr><td> dynamic viscosity <td> Poise
     * <tr><td> charge            <td> multiple of electron charge (1.0 is a proton)
     * <tr><td> dipole            <td> charge*Angstroms
     * <tr><td> electric field    <td> volts/Angstrom
     * <tr><td> density           <td> gram/cm^dim
     * <tr>
     * </table>
     */
    REAL,
    /*!
     * \brief \b METAL style
     *
     * <table>
     * <caption id="multi_row">REAL style</caption>
     * <tr><th> Units             <th> Description
     * <tr><td> mass              <td> grams/mole
     * <tr><td> distance          <td> Angstroms
     * <tr><td> time              <td> pico-seconds
     * <tr><td> energy            <td> eV
     * <tr><td> velocity          <td> Angstroms/pico-second
     * <tr><td> force             <td> eV/Angstrom
     * <tr><td> torque            <td> eV
     * <tr><td> temperature       <td> Kelvin
     * <tr><td> pressure          <td> bars
     * <tr><td> dynamic viscosity <td> Poise
     * <tr><td> charge            <td> multiple of electron charge (1.0 is a proton)
     * <tr><td> dipole            <td> charge*Angstroms
     * <tr><td> electric field    <td> volts/Angstrom
     * <tr><td> density           <td> gram/cm^dim
     * <tr>
     * </table>
     */
    METAL,
    /*!
     * \brief \b SI style
     *
     * <table>
     * <caption id="multi_row">REAL style</caption>
     * <tr><th> Units             <th> Description
     * <tr><td> mass              <td> kilograms
     * <tr><td> distance          <td> meters
     * <tr><td> time              <td> seconds
     * <tr><td> energy            <td> Joules
     * <tr><td> velocity          <td> meters/second
     * <tr><td> force             <td> Newtons
     * <tr><td> torque            <td> Newton-meters
     * <tr><td> temperature       <td> Kelvin
     * <tr><td> pressure          <td> Pascals
     * <tr><td> dynamic viscosity <td> Pascal*second
     * <tr><td> charge            <td> Coulombs (1.6021765e-19 is a proton)
     * <tr><td> dipole            <td> Coulombs*meters
     * <tr><td> electric field    <td> volts/meter
     * <tr><td> density           <td> kilograms/meter^dim
     * <tr>
     * </table>
     */
    SI,
    /*!
     * \brief \b CGS style
     *
     * <table>
     * <caption id="multi_row">REAL style</caption>
     * <tr><th> Units             <th> Description
     * <tr><td> mass              <td> grams
     * <tr><td> distance          <td> centimeters
     * <tr><td> time              <td> seconds
     * <tr><td> energy            <td> ergs
     * <tr><td> velocity          <td> centimeters/second
     * <tr><td> force             <td> dynes
     * <tr><td> torque            <td> dyne-centimeters
     * <tr><td> temperature       <td> Kelvin
     * <tr><td> pressure          <td> dyne/cm^2 or barye = 1.0e-6 bars
     * <tr><td> dynamic viscosity <td> Poise
     * <tr><td> charge            <td> statcoulombs or esu (4.8032044e-10 is a proton)
     * <tr><td> dipole            <td> stat-coul-cm = 10^18 debye
     * <tr><td> electric field    <td> statvolt/cm or dyne/esu
     * <tr><td> density           <td> grams/cm^dim
     * <tr>
     * </table>
     */
    CGS,
    /*!
     * \brief \b ELECTRON style
     *
     * <table>
     * <caption id="multi_row">REAL style</caption>
     * <tr><th> Units             <th> Description
     * <tr><td> mass              <td> atomic mass units
     * <tr><td> distance          <td> Bohr
     * <tr><td> time              <td> femto-seconds
     * <tr><td> energy            <td> Hartrees
     * <tr><td> velocity          <td> Bohr/atomic time units [1.03275e-15 seconds]
     * <tr><td> force             <td> Hartrees/Bohr
     * <tr><td> temperature       <td> Kelvin
     * <tr><td> pressure          <td> Pascals
     * <tr><td> charge            <td> multiple of electron charge (1.0 is a proton)
     * <tr><td> dipole            <td> Debye
     * <tr><td> electric field    <td> volts/cm
     * <tr>
     * </table>
     */
    ELECTRON
};

/*!
 * \enum LatticeType
 * \ingroup Units_Module
 *
 * \brief The LatticeType
 */
enum class LatticeType
{
    /*! */
    NONE,
    /*! Simple Cubic Lattice */
    SC,
    /*! Body-Centred Cubic Lattice */
    BCC,
    /*! Face-Centred Cubic Lattice */
    FCC,
    /*! Hexagonal Close-Packed Lattice */
    HCP,
    /*!  Lattice */
    DIAMOND,
    /*!  Lattice */
    SQ,
    /*!  Lattice */
    SQ2,
    /*!  Lattice */
    HEX,
    /*!  Lattice */
    CUSTOM
};

/*! \fn std::string getUnitStyleName(UnitStyle const &style)
 *
 * \brief Get the Unit Style Name object
 * \ingroup Units_Module
 *
 * \param style  The unit system style \sa umuq::UnitStyle
 *
 * \return std::string The unit name in string format
 */
inline std::string getUnitStyleName(UnitStyle const &style)
{
    // determine unit system and set lmps_units flag
    std::string STYLE = (style == UnitStyle::REAL) ? "REAL" : (style == UnitStyle::METAL) ? "METAL" : (style == UnitStyle::SI) ? "SI" : (style == UnitStyle::CGS) ? "CGS" : (style == UnitStyle::ELECTRON) ? "ELECTRON" : "";
    return STYLE;
}

/*! \class units
 * \ingroup Units_Module
 *
 * \brief This is a class which creates units of the system
 *
 * Working with different simulations codes one can create a units of the system
 * which working with and convert from other units to this.
 *
 * \todo
 * Currently only \c UnitStyle of METAL and conversion from \c ELECTRON are supported.
 * It should be extended for all available systems
 */
class units
{
public:
    /*!
     * \brief Construct a new units object
     *
     * The default unit system is [METAL](umuq::UnitStyle::METAL)
     */
    units();

    /*!
     * \brief Construct a new units object
     *
     * \param style  The unit system style \sa umuq::UnitStyle
     */
    explicit units(UnitStyle const &style);

    /*!
     * \brief Construct a new units object
     *
     * \param style  The unit system style \sa umuq::UnitStyle
     */
    units(std::string const &style);

    /*!
     * \brief Destroy the units object
     *
     */
    ~units();

    /*!
     * \brief Move constructor, construct a new units object
     *
     * \param other units object
     */
    explicit units(units &&other);

    /*!
     * \brief Move assignment operator
     *
     * \param other units object
     *
     * \returns units& units object
     */
    units &operator=(units &&other);

    /*!
     * \brief Get the Unit Style object
     *
     * \return UnitStyle
     */
    inline UnitStyle getUnitStyle();

    /*!
     * \brief Get the Unit Style Name object
     *
     * \return std::string
     */
    inline std::string getUnitStyleName();

    /*!
     * \brief Convert the input style (\c fromStyle) to the constrcuted style
     *
     * \param fromStyle The unit system style which needs to be converted to the current constrcuted style \sa umuq::UnitStyle
     */
    bool convertFromStyle(UnitStyle const &fromStyle);

    /*!
     * \brief Convert the input style (\c fromStyle) to the constrcuted style
     *
     * \param fromStyle The unit system style which needs to be converted to the current constrcuted style \sa umuq::UnitStyle
     */
    bool convertFromStyle(std::string const &fromStyle);

    /*!
     * \brief Convert the constrcuted style to the input style (\c toStyle)
     *
     * \param toStyle The unit system style which the current constrcuted style needs to be converted \sa umuq::UnitStyle
     */
    bool convertToStyle(UnitStyle const &toStyle);

    /*!
     * \brief Convert the constrcuted style to the input style (\c toStyle)
     *
     * \param toStyle The unit system style which the current constrcuted style needs to be converted to that \sa umuq::UnitStyle
     */
    bool convertToStyle(std::string const &toStyle);

    /*!
     * \brief Convert length
     *
     * \param value Input length
     * \note
     * \c units::convertFromStyle or \c units::convertToStyle should be called before calling this routine otherwise nothing will be changed
     */
    inline void convertLength(double &value);
    inline void convertLength(std::vector<double> &value);
    inline void convertLength(double *value, int const nSize);

    /*!
     * \brief Convert energy
     *
     * \param value Input energy
     * \note
     * \c units::convertFromStyle or \c units::convertToStyle should be called before calling this routine otherwise nothing will be changed
     */
    inline void convertEnergy(double &value);
    inline void convertEnergy(std::vector<double> &value);
    inline void convertEnergy(double *value, int const nSize);

    /*!
     * \brief Convert charge
     *
     * \param value Input charge
     * \note
     * \c units::convertFromStyle or \c units::convertToStyle should be called before calling this routine otherwise nothing will be changed
     */
    inline void convertCharge(double &value);
    inline void convertCharge(std::vector<double> &value);
    inline void convertCharge(double *value, int const nSize);

    /*!
     * \brief Convert temperature
     *
     * \param value Input temperature
     * \note
     * \c units::convertFromStyle or \c units::convertToStyle should be called before calling this routine otherwise nothing will be changed
     */
    inline void convertTemperature(double &value);
    inline void convertTemperature(std::vector<double> &value);
    inline void convertTemperature(double *value, int const nSize);

    /*!
     * \brief Convert time
     *
     * \param value Input time
     * \note
     * \c units::convertFromStyle or \c units::convertToStyle should be called before calling this routine otherwise nothing will be changed
     */
    inline void convertTime(double &value);
    inline void convertTime(std::vector<double> &value);
    inline void convertTime(double *value, int const nSize);

    /*!
     * \brief Convert force
     *
     * \param value Input force
     * \note
     * \c units::convertFromStyle or \c units::convertToStyle should be called before calling this routine otherwise nothing will be changed
     */
    inline void convertForce(double &value);
    inline void convertForce(std::vector<double> &value);
    inline void convertForce(double *value, int const nSize);

private:
    /*!
     * \brief Delete a units object copy construction
     *
     * Avoiding implicit generation of the copy constructor.
     */
    units(units const &) = delete;

    /*!
     * \brief Delete a units object assignment
     *
     * Avoiding implicit copy assignment.
     */
    units &operator=(units const &) = delete;

protected:
    /*! The length scaling factor */
    double lengthUnitScale;
    /*! The energy scaling factor */
    double energyUnitScale;
    /*! The charge scaling factor */
    double chargeUnitScale;
    /*! The temperature scaling factor */
    double temperatureUnitScale;
    /*! The time scaling factor */
    double timeUnitScale;
    /*! The force scaling factor */
    double forceUnitScale;

private:
    // Pressure:
    /*! \f$ 1 \frac{eV}{Angstrom^3} = 160.21766208 GPa \f$ */
    static constexpr double const eV_Angstrom3TGPa = 160.21766208;
    /*! \f$ 1 \frac{Hartree}{Bohr^3} = 29421.02648438959 GPa \f$ */
    static constexpr double const Hartree_Bohr3TGPa = 29421.02648438959;
    /*! \f$ 1 GPa = 145037.738007218 \frac{pound}{square inch} \f$ */
    static constexpr double const GPaTpound_inch2 = 145037.738007218;
    /*! \f$ 1 atm = 1.01325 bar \f$ */
    static constexpr double const atmTbar = 1.01325;
    /*! \f$ 1 atm =  0.000101325 GPa \f$ */
    static constexpr double const atmTGPa = 0.000101325;
    /*! \f$ 1 pascal = 1.0E-09 GPa \f$ */
    static constexpr double const pascalTGPa = 1.0E-9;

    // Force:
    /*! \f$ 1 \frac{Rydberg}{Bohr} = 25.71104309541616 \frac{eV}{Angstrom}, \f$ where Ry is Rydberg unit of energy. */
    static constexpr double const Rydberg_BohrTeV_Angstrom = 25.71104309541616;
    /*! \f$ 1 \frac{Hartree}{Bohr} = 51.42208619083232 \frac{eV}{Angstrom} \f$ */
    static constexpr double const Hartree_BohrTeV_Angstrom = 51.42208619083232;

    // Energy:
    /*! \f$ 1 Hartree = 2 Rydberg = 27.211396 eV \f$ */
    static constexpr double const HartreeTRydberg = 2;
    /*! \f$ 1 Hartree = 27.211396 eV \f$ */
    static constexpr double const HartreeTeV = 27.211396;
    /*! \f$ 1 \frac{kJ}{mol} = 0.0103642688 \frac{eV}{atom} \f$ */
    static constexpr double const kJ_molTeV_atom = 0.0103642688;
    /*! \f$ 1 Joule = 6.24150965x10^(18) eV (CODATA) \f$ */
    static constexpr double const JouleTeV = 6.24150965E18;
    /*! \f$ 1 eV = 1.6021766208*10^(-19) Joules \f$ */
    static constexpr double const eVTJoules = 1.6021766208E-19;

    // Length:
    /*! \f$ 1 Bohr = 0.529177208 Angstrom \f$ */
    static constexpr double const BohrTAngstrom = 0.529177208;

    // Time:
    static constexpr double const psTfs = 1.0E03;
    static constexpr double const fsTs = 1.0E-15;
    static constexpr double const fsTps = 1.0E-03;

    // Mass:
    /*! 1 atomic mass unit */
    static constexpr double const amuTkg = 1.660540E-27;

private:
    /*! System unit style */
    UnitStyle unitStyle;
    /*! Length unit of the specified system style */
    LengthUnit lengthUnit;
    /*! Energy unit of the specified system style */
    EnergyUnit energyUnit;
    /*! Charge unit of the specified system style */
    ChargeUnit chargeUnit;
    /*! Temperature unit of the specified system style */
    TemperatureUnit temperatureUnit;
    /*! Time unit of the specified system style */
    TimeUnit timeUnit;
    /*! Force unit of the specified system style */
    ForceUnit forceUnit;
};

units::units() : unitStyle(UnitStyle::METAL),
                 lengthUnit(LengthUnit::A),
                 energyUnit(EnergyUnit::eV),
                 chargeUnit(ChargeUnit::electron),
                 temperatureUnit(TemperatureUnit::K),
                 timeUnit(TimeUnit::ps),
                 forceUnit(ForceUnit::eV_Angstrom),
                 lengthUnitScale(1.0),
                 energyUnitScale(1.0),
                 chargeUnitScale(1.0),
                 temperatureUnitScale(1.0),
                 timeUnitScale(1.0),
                 forceUnitScale(1.0) {}

units::units(UnitStyle const &style)
{
    // determine unit system and set lmps_units flag
    switch (style)
    {
    case UnitStyle::REAL:
        unitStyle = UnitStyle::REAL;
        lengthUnit = LengthUnit::A;
        energyUnit = EnergyUnit::kcal_mol;
        chargeUnit = ChargeUnit::electron;
        temperatureUnit = TemperatureUnit::K;
        timeUnit = TimeUnit::fs;
        forceUnit = ForceUnit::Kcal_moleAngstrom;
        break;
    case UnitStyle::METAL:
        unitStyle = UnitStyle::METAL;
        lengthUnit = LengthUnit::A;
        energyUnit = EnergyUnit::eV;
        chargeUnit = ChargeUnit::electron;
        temperatureUnit = TemperatureUnit::K;
        timeUnit = TimeUnit::ps;
        forceUnit = ForceUnit::eV_Angstrom;
        break;
    case UnitStyle::SI:
        unitStyle = UnitStyle::SI;
        lengthUnit = LengthUnit::m;
        energyUnit = EnergyUnit::J;
        chargeUnit = ChargeUnit::Coulomb;
        temperatureUnit = TemperatureUnit::K;
        timeUnit = TimeUnit::s;
        forceUnit = ForceUnit::Newtons;
        break;
    case UnitStyle::CGS:
        unitStyle = UnitStyle::CGS;
        lengthUnit = LengthUnit::cm;
        energyUnit = EnergyUnit::erg;
        chargeUnit = ChargeUnit::statCoulomb;
        temperatureUnit = TemperatureUnit::K;
        timeUnit = TimeUnit::s;
        forceUnit = ForceUnit::dynes;
        break;
    case UnitStyle::ELECTRON:
        unitStyle = UnitStyle::ELECTRON;
        lengthUnit = LengthUnit::Bohr;
        energyUnit = EnergyUnit::Hartree;
        chargeUnit = ChargeUnit::electron;
        temperatureUnit = TemperatureUnit::K;
        timeUnit = TimeUnit::fs;
        forceUnit = ForceUnit::Hartrees_Bohr;
        break;
    default:
        UMUQFAIL("Unknown unit style by UMUQ!");
        break;
    }

    lengthUnitScale = 1.0;
    energyUnitScale = 1.0;
    chargeUnitScale = 1.0;
    temperatureUnitScale = 1.0;
    timeUnitScale = 1.0;
    forceUnitScale = 1.0;
}

units::units(std::string const &style)
{
    umuq::parser p;
    auto STYLE = p.toupper(style);

    // determine unit system style and set the flags
    if (STYLE == "REAL")
    {
        unitStyle = UnitStyle::REAL;
        lengthUnit = LengthUnit::A;
        energyUnit = EnergyUnit::kcal_mol;
        chargeUnit = ChargeUnit::electron;
        temperatureUnit = TemperatureUnit::K;
        timeUnit = TimeUnit::fs;
        forceUnit = ForceUnit::Kcal_moleAngstrom;
    }
    else if (STYLE == "METAL")
    {
        unitStyle = UnitStyle::METAL;
        lengthUnit = LengthUnit::A;
        energyUnit = EnergyUnit::eV;
        chargeUnit = ChargeUnit::electron;
        temperatureUnit = TemperatureUnit::K;
        timeUnit = TimeUnit::ps;
        forceUnit = ForceUnit::eV_Angstrom;
    }
    else if (STYLE == "SI")
    {
        unitStyle = UnitStyle::SI;
        lengthUnit = LengthUnit::m;
        energyUnit = EnergyUnit::J;
        chargeUnit = ChargeUnit::Coulomb;
        temperatureUnit = TemperatureUnit::K;
        timeUnit = TimeUnit::s;
        forceUnit = ForceUnit::Newtons;
    }
    else if (STYLE == "CGS")
    {
        unitStyle = UnitStyle::CGS;
        lengthUnit = LengthUnit::cm;
        energyUnit = EnergyUnit::erg;
        chargeUnit = ChargeUnit::statCoulomb;
        temperatureUnit = TemperatureUnit::K;
        timeUnit = TimeUnit::s;
        forceUnit = ForceUnit::dynes;
    }
    else if (STYLE == "ELECTRON")
    {
        unitStyle = UnitStyle::ELECTRON;
        lengthUnit = LengthUnit::Bohr;
        energyUnit = EnergyUnit::Hartree;
        chargeUnit = ChargeUnit::electron;
        temperatureUnit = TemperatureUnit::K;
        timeUnit = TimeUnit::fs;
        forceUnit = ForceUnit::Hartrees_Bohr;
    }
    else
    {
        UMUQFAIL("The unit style of (", STYLE, ") is unknown by UMUQ!");
    }

    lengthUnitScale = 1.0;
    energyUnitScale = 1.0;
    chargeUnitScale = 1.0;
    temperatureUnitScale = 1.0;
    timeUnitScale = 1.0;
    forceUnitScale = 1.0;
}

units::~units() {}

units::units(units &&other)
{
    unitStyle = other.unitStyle;
    lengthUnit = other.lengthUnit;
    energyUnit = other.energyUnit;
    chargeUnit = other.chargeUnit;
    temperatureUnit = other.temperatureUnit;
    timeUnit = other.timeUnit;
    forceUnit = other.forceUnit;
    lengthUnitScale = other.lengthUnitScale;
    energyUnitScale = other.energyUnitScale;
    chargeUnitScale = other.chargeUnitScale;
    temperatureUnitScale = other.temperatureUnitScale;
    timeUnitScale = other.timeUnitScale;
    forceUnitScale = other.forceUnitScale;
}

units &units::operator=(units &&other)
{
    unitStyle = other.unitStyle;
    lengthUnit = other.lengthUnit;
    energyUnit = other.energyUnit;
    chargeUnit = other.chargeUnit;
    temperatureUnit = other.temperatureUnit;
    timeUnit = other.timeUnit;
    forceUnit = other.forceUnit;
    lengthUnitScale = other.lengthUnitScale;
    energyUnitScale = other.energyUnitScale;
    chargeUnitScale = other.chargeUnitScale;
    temperatureUnitScale = other.temperatureUnitScale;
    timeUnitScale = other.timeUnitScale;
    forceUnitScale = other.forceUnitScale;

    return *this;
}

inline UnitStyle units::getUnitStyle() { return unitStyle; }

inline std::string units::getUnitStyleName() { return umuq::getUnitStyleName(unitStyle); }

bool units::convertFromStyle(UnitStyle const &fromStyle)
{
    if (fromStyle == unitStyle)
    {
        UMUQWARNING("The old style == the current style == ", umuq::getUnitStyleName(unitStyle), " and nothing will change!");
        return true;
    }
    if (fromStyle == UnitStyle::ELECTRON)
    {
        if (unitStyle == UnitStyle::METAL)
        {
            lengthUnitScale = BohrTAngstrom;
            energyUnitScale = HartreeTeV;
            chargeUnitScale = 1.0;
            temperatureUnitScale = 1.0;
            timeUnitScale = fsTps;
            forceUnitScale = Hartree_BohrTeV_Angstrom;
            return true;
        }
    }
    UMUQFAILRETURN("Conversion from style (", umuq::getUnitStyleName(fromStyle), ") to the current constructed style (", umuq::getUnitStyleName(unitStyle), ") is not implemented!");
}

bool units::convertFromStyle(std::string const &fromStyle)
{
    umuq::parser p;
    auto STYLE = p.toupper(fromStyle);

    // determine unit system fromStyle
    if (STYLE == "REAL")
    {
        return convertFromStyle(UnitStyle::REAL);
    }
    else if (STYLE == "METAL")
    {
        return convertFromStyle(UnitStyle::METAL);
    }
    else if (STYLE == "SI")
    {
        return convertFromStyle(UnitStyle::SI);
    }
    else if (STYLE == "CGS")
    {
        return convertFromStyle(UnitStyle::CGS);
    }
    else if (STYLE == "ELECTRON")
    {
        return convertFromStyle(UnitStyle::ELECTRON);
    }

    UMUQFAILRETURN("The unit style of (", STYLE, ") is unknown by UMUQ!");
}

bool units::convertToStyle(UnitStyle const &toStyle)
{
    if (toStyle == unitStyle)
    {
        UMUQWARNING("The old style == the current style == ", umuq::getUnitStyleName(unitStyle), " and nothing will change!");
        return true;
    }
    if (toStyle == UnitStyle::METAL)
    {
        if (unitStyle == UnitStyle::ELECTRON)
        {
            lengthUnitScale = BohrTAngstrom;
            energyUnitScale = HartreeTeV;
            chargeUnitScale = 1.0;
            temperatureUnitScale = 1.0;
            timeUnitScale = fsTps;
            forceUnitScale = Hartree_BohrTeV_Angstrom;
            return true;
        }
    }

    UMUQFAILRETURN("Conversion from the current constructed style (", umuq::getUnitStyleName(unitStyle), ") to the new style (", umuq::getUnitStyleName(toStyle), ") is not implemented!");
}

bool units::convertToStyle(std::string const &toStyle)
{
    umuq::parser p;
    auto STYLE = p.toupper(toStyle);

    // determine unit system style
    if (STYLE == "REAL")
    {
        return convertToStyle(UnitStyle::REAL);
    }
    else if (STYLE == "METAL")
    {
        return convertToStyle(UnitStyle::METAL);
    }
    else if (STYLE == "SI")
    {
        return convertToStyle(UnitStyle::SI);
    }
    else if (STYLE == "CGS")
    {
        return convertToStyle(UnitStyle::CGS);
    }
    else if (STYLE == "ELECTRON")
    {
        return convertToStyle(UnitStyle::ELECTRON);
    }

    UMUQFAILRETURN("The unit style of (", STYLE, ") is unknown by UMUQ!");
}

inline void units::convertLength(double &value) { value *= lengthUnitScale; }

inline void units::convertLength(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= lengthUnitScale; });
}

inline void units::convertLength(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= lengthUnitScale; });
}

inline void units::convertEnergy(double &value) { value *= energyUnitScale; }

inline void units::convertEnergy(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= energyUnitScale; });
}

inline void units::convertEnergy(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= energyUnitScale; });
}

inline void units::convertCharge(double &value) { value *= chargeUnitScale; }

inline void units::convertCharge(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= chargeUnitScale; });
}

inline void units::convertCharge(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= chargeUnitScale; });
}

inline void units::convertTemperature(double &value) { value *= temperatureUnitScale; }

inline void units::convertTemperature(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= temperatureUnitScale; });
}

inline void units::convertTemperature(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= temperatureUnitScale; });
}

inline void units::convertTime(double &value) { value *= timeUnitScale; }

inline void units::convertTime(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= timeUnitScale; });
}

inline void units::convertTime(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= timeUnitScale; });
}

inline void units::convertForce(double &value) { value *= forceUnitScale; }

inline void units::convertForce(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= forceUnitScale; });
}

inline void units::convertForce(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= forceUnitScale; });
}

/*! \fn bool convert(std::vector<double> &value, UnitStyle const &fromStyle, UnitStyle const &toStyle)
 *
 * \ingroup Units_Module
 * \brief this is a general convert functionality
 *
 * \tparam UNIT the physical unit which we want to convert, it could be any of :
 * \b EnergyUnit \sa umuq::EnergyUnit
 * \b LengthUnit \sa umuq::LengthUnit
 * \b ChargeUnit \sa umuq::ChargeUnit
 * \b TemperatureUnit \sa umuq::TemperatureUnit
 * \b TimeUnit \sa umuq::TimeUnit
 * \b ForceUnit \sa umuq::ForceUnit
 *
 * \param value      Array of values which we want to convert from \c fromStyle to \c toStyle style
 * \param fromStyle  The input style
 * \param toStyle    The output style
 *
 * \return true
 * \return false
 */
template <class UNIT>
bool convert(std::vector<double> &value, UnitStyle const &fromStyle, UnitStyle const &toStyle)
{
    UMUQFAILRETURN("This is not implemented on purpose!");
}

template <>
bool convert<umuq::LengthUnit>(std::vector<double> &value, UnitStyle const &fromStyle, UnitStyle const &toStyle)
{
    if (fromStyle == toStyle)
    {
        UMUQWARNING("The old style == the current style == ", umuq::getUnitStyleName(fromStyle), " and nothing will change!");
        return true;
    }
    umuq::units u(toStyle);
    if (u.convertFromStyle(fromStyle))
    {
        u.convertLength(value);
        return true;
    }
    return false;
}

template <>
bool convert<umuq::EnergyUnit>(std::vector<double> &value, UnitStyle const &fromStyle, UnitStyle const &toStyle)
{
    if (fromStyle == toStyle)
    {
        UMUQWARNING("The old style == the current style == ", umuq::getUnitStyleName(fromStyle), " and nothing will change!");
        return true;
    }
    umuq::units u(toStyle);
    if (u.convertFromStyle(fromStyle))
    {
        u.convertEnergy(value);
        return true;
    }
    return false;
}

template <>
bool convert<umuq::ChargeUnit>(std::vector<double> &value, UnitStyle const &fromStyle, UnitStyle const &toStyle)
{
    if (fromStyle == toStyle)
    {
        UMUQWARNING("The old style == the current style == ", umuq::getUnitStyleName(fromStyle), " and nothing will change!");
        return true;
    }
    umuq::units u(toStyle);
    if (u.convertFromStyle(fromStyle))
    {
        u.convertCharge(value);
        return true;
    }
    return false;
}

template <>
bool convert<umuq::TemperatureUnit>(std::vector<double> &value, UnitStyle const &fromStyle, UnitStyle const &toStyle)
{
    if (fromStyle == toStyle)
    {
        UMUQWARNING("The old style == the current style == ", umuq::getUnitStyleName(fromStyle), " and nothing will change!");
        return true;
    }
    umuq::units u(toStyle);
    if (u.convertFromStyle(fromStyle))
    {
        u.convertTemperature(value);
        return true;
    }
    return false;
}

template <>
bool convert<umuq::TimeUnit>(std::vector<double> &value, UnitStyle const &fromStyle, UnitStyle const &toStyle)
{
    if (fromStyle == toStyle)
    {
        UMUQWARNING("The old style == the current style == ", umuq::getUnitStyleName(fromStyle), " and nothing will change!");
        return true;
    }
    umuq::units u(toStyle);
    if (u.convertFromStyle(fromStyle))
    {
        u.convertTime(value);
        return true;
    }
    return false;
}

template <>
bool convert<umuq::ForceUnit>(std::vector<double> &value, UnitStyle const &fromStyle, UnitStyle const &toStyle)
{
    if (fromStyle == toStyle)
    {
        UMUQWARNING("The old style == the current style == ", umuq::getUnitStyleName(fromStyle), " and nothing will change!");
        return true;
    }
    umuq::units u(toStyle);
    if (u.convertFromStyle(fromStyle))
    {
        u.convertForce(value);
        return true;
    }
    return false;
}

/*! \fn bool convert(std::vector<double> &value,  std::string const &fromStyle,  std::string const &toStyle)
 *
 * \ingroup Units_Module
 * \brief this is a general convert functionality
 *
 * \tparam UNIT the physical unit which we want to convert, it could be any of :
 * \b EnergyUnit \sa umuq::EnergyUnit
 * \b LengthUnit \sa umuq::LengthUnit
 * \b ChargeUnit \sa umuq::ChargeUnit
 * \b TemperatureUnit \sa umuq::TemperatureUnit
 * \b TimeUnit \sa umuq::TimeUnit
 *
 * \param value      Array of values which we want to convert from \c fromStyle to \c toStyle style
 * \param fromStyle  The input style
 * \param toStyle    The output style
 *
 * \return true
 * \return false
 */
template <class UNIT>
bool convert(std::vector<double> &value, std::string const &fromStyle, std::string const &toStyle)
{
    UMUQFAILRETURN("This is not implemented on purpose!");
}

template <>
bool convert<umuq::LengthUnit>(std::vector<double> &value, std::string const &fromStyle, std::string const &toStyle)
{
    umuq::parser p;
    auto FromStyle = p.toupper(fromStyle);
    auto ToStyle = p.toupper(toStyle);
    if (FromStyle == ToStyle)
    {
        UMUQWARNING("The old style == the current style == ", FromStyle, " and nothing will change!");
        return true;
    }
    umuq::units u(ToStyle);
    if (u.convertFromStyle(FromStyle))
    {
        u.convertLength(value);
        return true;
    }
    return false;
}

template <>
bool convert<umuq::EnergyUnit>(std::vector<double> &value, std::string const &fromStyle, std::string const &toStyle)
{
    umuq::parser p;
    auto FromStyle = p.toupper(fromStyle);
    auto ToStyle = p.toupper(toStyle);
    if (FromStyle == ToStyle)
    {
        UMUQWARNING("The old style == the current style == ", FromStyle, " and nothing will change!");
        return true;
    }
    umuq::units u(ToStyle);
    if (u.convertFromStyle(FromStyle))
    {
        u.convertEnergy(value);
        return true;
    }
    return false;
}

template <>
bool convert<umuq::ChargeUnit>(std::vector<double> &value, std::string const &fromStyle, std::string const &toStyle)
{
    umuq::parser p;
    auto FromStyle = p.toupper(fromStyle);
    auto ToStyle = p.toupper(toStyle);
    if (FromStyle == ToStyle)
    {
        UMUQWARNING("The old style == the current style == ", FromStyle, " and nothing will change!");
        return true;
    }
    umuq::units u(ToStyle);
    if (u.convertFromStyle(FromStyle))
    {
        u.convertCharge(value);
        return true;
    }
    return false;
}

template <>
bool convert<umuq::TemperatureUnit>(std::vector<double> &value, std::string const &fromStyle, std::string const &toStyle)
{
    umuq::parser p;
    auto FromStyle = p.toupper(fromStyle);
    auto ToStyle = p.toupper(toStyle);
    if (FromStyle == ToStyle)
    {
        UMUQWARNING("The old style == the current style == ", FromStyle, " and nothing will change!");
        return true;
    }
    umuq::units u(ToStyle);
    if (u.convertFromStyle(FromStyle))
    {
        u.convertTemperature(value);
        return true;
    }
    return false;
}

template <>
bool convert<umuq::TimeUnit>(std::vector<double> &value, std::string const &fromStyle, std::string const &toStyle)
{
    umuq::parser p;
    auto FromStyle = p.toupper(fromStyle);
    auto ToStyle = p.toupper(toStyle);
    if (FromStyle == ToStyle)
    {
        UMUQWARNING("The old style == the current style == ", FromStyle, " and nothing will change!");
        return true;
    }
    umuq::units u(ToStyle);
    if (u.convertFromStyle(FromStyle))
    {
        u.convertTime(value);
        return true;
    }
    return false;
}

template <>
bool convert<umuq::ForceUnit>(std::vector<double> &value, std::string const &fromStyle, std::string const &toStyle)
{
    umuq::parser p;
    auto FromStyle = p.toupper(fromStyle);
    auto ToStyle = p.toupper(toStyle);
    if (FromStyle == ToStyle)
    {
        UMUQWARNING("The old style == the current style == ", FromStyle, " and nothing will change!");
        return true;
    }
    umuq::units u(ToStyle);
    if (u.convertFromStyle(FromStyle))
    {
        u.convertForce(value);
        return true;
    }
    return false;
}

/*! \fn void convertFractionalToCartesianCoordinates(std::vector<double> const &boundingVectors, std::vector<double> &fractionalCoordinates)
 * \ingroup Units_Module
 *
 * \brief Function to convert fractional species coordinates to cartesian coordinates
 *
 * \param boundingVectors        Definition using parallelepiped
 * \param fractionalCoordinates  Fractional species coordinates on input and cartesian coordinates on return
 */
void convertFractionalToCartesianCoordinates(std::vector<double> const &boundingVectors, std::vector<double> &fractionalCoordinates)
{
    // Creat an instance of a lattice object with the input bounding vectors
    umuq::lattice l(boundingVectors);
    fractionalCoordinates = l.fractionalToCartesian(fractionalCoordinates);
}

} // namespace umuq

#endif // UMUQ_UNITS