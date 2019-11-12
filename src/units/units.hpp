#ifndef UMUQ_UNITS_H
#define UMUQ_UNITS_H

#include "core/core.hpp"
#include "misc/parser.hpp"

#include <string>
#include <vector>
#include <algorithm>
#include <map>

namespace umuq
{

/*!
 * \defgroup Units_Module Units module
 *
 * This is the Units Module of %UMUQ providing all necessary classes
 * for physical units and their conversion currently supported in %UMUQ.
 */

/*!
 * \enum UnitType
 * \ingroup Units_Module
 *
 * \brief The UnitType class
 *
 */
enum class UnitType
{
    Mass,
    Length,
    Time,
    Energy,
    Velocity,
    Force,
    Torque,
    Temperature,
    Pressure,
    Viscosity,
    Charge,
    Dipole,
    ElectricField,
    Density
};

/*!
 * \enum MassUnit
 * \ingroup Units_Module
 *
 * \brief The MassUnit class
 *
 */
enum class MassUnit
{
    /*! The Gram unit of mass. */
    Gram,
    /*! The Kilo Gram unit of mass. */
    kGram,
    /*! The Picogram unit of mass. */
    Picogram,
    /*! The Attogram unit of mass. */
    Attogram,
    /*! The Gram per mole unit of mass. */
    Gram_Mol,
    /*! The Atomic mass unit (molecular weight). */
    Amu
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
    /*! The Bohr unit of length.*/
    Bohr,
    /*! The meter unit of length. */
    Meter,
    /*! The centimeter unit of length.*/
    Centimeter,
    /*! The micro-meter unit of length. */
    Micrometer,
    /*! The nano-meter unit of length. */
    Nonometer,
    /*! The angstrom unit of length. */
    Angstrom
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
    /*! The Atomic time unit. */
    Atu,
    /*! The Atomic time unit used in Electron system. */
    AtuElectron,
    /*! The second unit of time. */
    Second,
    /*! The micro-second unit of time. */
    Microsecond,
    /*! The nano-second unit of time. */
    Nanosecond,
    /*! The pico-second unit of time. */
    Picosecond,
    /*! The femto-second unit of time. */
    Femtosecond
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
    /*! The Erg unit of energy. */
    Erg,
    /*! The \f$ \text{Dyne}\times\text{Centimeter} unit of energy. */
    DyneCentimeter,
    /*! The Electron-volt unit of energy. */
    ElectronVolt,
    /*! The Hartree unit of energy.*/
    Hartree,
    /*! The Joule unit of energy. */
    Joule,
    /*! The kilocalorie unit of energy. */
    kCal,
    /*! The \f$ \frac{\text{kilocalorie}}{\text{mole}} \f$ unit of energy. */
    kCal_mol,
    /*! The \f$ {\frac{\text{amu}~\text{Angstrom}^2}{\text{Picosecond}^2}} \f$ unit of energy. */
    AmuA2_Picosecond2,
    /*! The \f$ {\frac{\text{Picogram}~\text{Micrometer}^2}{\text{Microsecond}^2}} \f$ unit of energy. */
    PicogramMicrometer2_Microsecond2,
    /*! The \f$ {\frac{\text{Attogram}~\text{Nanometer}^2}{\text{Nanosecond}^2}} \f$ unit of energy. */
    AttogramNanometer2_Nanosecond2,
    /*! The \f$ \text{Newton}\times\text{Meter} unit of energy. */
    Newton_Meter
};

enum class TorqueUnit
{
    /*! The \f$ \text{Dyne}\times\text{Centimeter} unit of energy. */
    DyneCentimeter,
    /*! The Electron-volt unit of energy. */
    ElectronVolt,
    /*! The Hartree unit of energy.*/
    Hartree,
    /*! The \f$ \frac{\text{kilocalorie}}{\text{mole}} \f$ unit of energy. */
    kCal_mol,
    /*! The \f$ \text{Newton}\times\text{Meter} unit of energy. */
    Newton_Meter
};

/*!
 * \enum VelocityUnit
 * \ingroup Units_Module
 *
 * \brief The VelocityUnit class
 *
 */
enum class VelocityUnit
{
    /*! The \f$ {\frac{\text{Meter}}{\text{Second}}} \f$ unit of velocity. */
    Meter_Second,
    /*! The \f$ {\frac{\text{Angstrom}}{\text{Femtosecond}}} \f$ unit of velocity. */
    Angstrom_Femtosecond,
    /*! The \f$ {\frac{\text{Angstrom}}{\text{Picosecond}}} \f$ unit of velocity. */
    Angstrom_Picosecond,
    /*! The \f$ {\frac{\text{Micrometer}}{\text{Microsecond}}} \f$ unit of velocity. */
    Micrometer_Microsecond,
    /*! The \f$ {\frac{\text{Nanometer}}{\text{Nanosecond}}} \f$ unit of velocity. */
    Nanometer_Nanosecond,
    /*! The \f$ {\frac{\text{Centimeter}}{\text{Second}}} \f$ unit of velocity. */
    Centimeter_Second,
    /*! The \f$ {\frac{\text{Bohr}}{\text{AtuElectron}}} \f$ unit of velocity. */
    Bohr_AtuElectron
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
    kCal_moleAngstrom,
    /*! The unit of force in \b METAL style */
    ElectronVolt_Angstrom,
    /*! The unit of force in \b SI style */
    Newton,
    /*! The unit of force in \b CGS style */
    Dyne,
    /*! The unit of force in \b ELECTRON style */
    Hartrees_Bohr,
    /*! The \f$ {\frac{\text{Picogram}~\text{Micrometer}}{\text{Microsecond}^2}} \f$ unit of force */
    PicogramMicrometer_Microsecond2,
    /*! The \f$ {\frac{\text{Attogram}~\text{Nanometer}}{\text{Nanosecond}^2}} \f$ unit of force */
    AttogramNanometer_Nanosecond2,
    /*! The unit of force */
    Rydberg_Bohr
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
    /*! The Kelvin unit of temperature. */
    Kelvin
};

/*!
 * \enum PressureUnit
 * \ingroup Units_Module
 *
 * \brief The PressureUnit class
 *
 */
enum class PressureUnit
{
    /*! The Pascal unit of pressure. */
    Pascal,
    /*! The Atmosphere unit of pressure. */
    Atmosphere,
    /*! The Bar unit of pressure. */
    Bar,
    /*! The \f$ \frac{\text{Dyne}}{\text{Centimeter}^2} \f$ unit of pressure. */
    Dyne_Centimeter2,
    /*! The \f$ \frac{\text{Picogram}}{\text{Micrometer}~\text{Microsecond}^2} \f$ unit of pressure. */
    Picogram_MicrometerMicrosecond2,
    /*! The \f$ \frac{\text{Attogram}}{\text{Nanometer}~\text{Nanosecond}^2} \f$ unit of pressure. */
    Attogram_NanometerNanosecond2
};

/*!
 * \enum ViscosityUnit
 * \ingroup Units_Module
 *
 * \brief The ViscosityUnit class
 *
 */
enum class ViscosityUnit
{
    /*! The SI unit of dynamic viscosity. */
    PascalSecond,
    /*! The Poise unit of dynamic viscosity. */
    Poise,
    /*! The \f$ \frac{\text{Amu}}{\text{Bohr}~\text{Femtosecond}} \f$ unit of dynamic viscosity. */
    Amu_BohrFemtosecond,
    /*! The \f$ \frac{\text{Picogram}}{\text{Micrometer}~\text{Microsecond}} \f$ unit of dynamic viscosity. */
    Picogram_MicrometerMicrosecond,
    /*! The \f$ \frac{\text{Attogram}}{\text{Nanometer}~\text{Nanosecond}} \f$ unit of dynamic viscosity. */
    Attogram_NanometerNanosecond
};

/*!
 * \enum ChargeUnit
 * \ingroup Units_Module
 *
 * \brief The ChargeUnit class
 *
 */
enum class ChargeUnit
{
    /*! The Coulomb unit of charge. */
    Coulomb,
    /*! The Electron unit of charge. */
    Electron,
    /*! The stat-coulomb unit of charge. */
    StatCoulomb,
    /*! The Pico-coulomb unit of charge. */
    PicoCoulomb
};

/*!
 * \enum DipoleUnit
 * \ingroup Units_Module
 *
 * \brief The DipoleUnit class
 *
 */
enum class DipoleUnit
{
    /*! The Coulomb-meter unit of dipole. */
    CoulombMeter,
    /*! The Electron-Angstrom unit of dipole. */
    ElectronAngstrom,
    /*! The stat-coulomb centimeter unit of dipole. */
    StatCoulombCentimeter,
    /*! The Debye unit of dipole. */
    Debye,
    /*! The Pico-coulomb micrometer unit of dipole. */
    PicoCoulombMicrometer,
    /*! The Electron nanometer unit of dipole. */
    ElectronNanometer
};

/*!
 * \enum ElectricFieldUnit
 * \ingroup Units_Module
 *
 * \brief The ElectricFieldUnit class
 *
 */
enum class ElectricFieldUnit
{
    /*! The \f$ \frac{\text{Volt}}{\text{Meter}} \f$ unit of Electric field. */
    Volt_Meter,
    /*! The \f$ \frac{\text{Volt}}{\text{Angstrom}} \f$ unit of Electric field. */
    Volt_Angstrom,
    /*! The \f$ \frac{\text{Stat-volt}}{\text{Centimeter}} \f$ unit of Electric field. */
    StatVolt_Centimeter,
    /*! The \f$ \frac{\text{Volt}}{\text{Centimeter}} \f$ unit of Electric field. */
    Volt_Centimeter,
    /*! The \f$ \frac{\text{Volt}}{\text{Micrometer}} \f$ unit of Electric field. */
    Volt_Micrometer,
    /*! The \f$ \frac{\text{Volt}}{\text{Nanometer}} \f$ unit of Electric field. */
    Volt_Nanometer
};

/*!
 * \enum DensityUnit
 * \ingroup Units_Module
 *
 * \brief The DensityUnit class
 *
 */
enum class DensityUnit
{
    /*! The \f$ \frac{\text{Gram}}{\text{Centimeter}^3} \f$ unit of density. */
    Gram_Centimeter3,
    /*! The \f$ \frac{\text{Amu}}{\text{Bohr}^3} \f$ unit of density. */
    Amu_Bohr3,
    /*! The \f$ \frac{\text{Kilogram}}{\text{Meter}^3} \f$ unit of density. */
    kGram_Meter3,
    /*! The \f$ \frac{\text{Picogram}}{\text{Micrometer}^3} \f$ unit of density. */
    Picogram_Micrometer3,
    /*! The \f$ \frac{\text{Attogram}}{\text{Nanometer}^3} \f$ unit of density. */
    Attogram_Nanometer3
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
     * <tr><td> charge            <td> multiple of Electron charge (1.0 is a proton)
     * <tr><td> dipole            <td> charge*Angstroms
     * <tr><td> electric field    <td> volts/Angstrom
     * <tr><td> density           <td> gram/Centimeter^dim
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
     * <tr><td> energy            <td> ElectronVolt
     * <tr><td> velocity          <td> Angstroms/pico-second
     * <tr><td> force             <td> ElectronVolt/Angstrom
     * <tr><td> torque            <td> ElectronVolt
     * <tr><td> temperature       <td> Kelvin
     * <tr><td> pressure          <td> bars
     * <tr><td> dynamic viscosity <td> Poise
     * <tr><td> charge            <td> multiple of Electron charge (1.0 is a proton)
     * <tr><td> dipole            <td> charge*Angstroms
     * <tr><td> electric field    <td> volts/Angstrom
     * <tr><td> density           <td> gram/Centimeter^dim
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
     * <tr><td> force             <td> Newton
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
     * <tr><td> force             <td> Dyne
     * <tr><td> torque            <td> dyne-centimeters
     * <tr><td> temperature       <td> Kelvin
     * <tr><td> pressure          <td> dyne/Centimeter^2 or barye = 1.0e-6 bars
     * <tr><td> dynamic viscosity <td> Poise
     * <tr><td> charge            <td> statcoulombs or esu (4.8032044e-10 is a proton)
     * <tr><td> dipole            <td> stat-coul-Centimeter = 10^18 debye
     * <tr><td> electric field    <td> statvolt/Centimeter or dyne/esu
     * <tr><td> density           <td> grams/Centimeter^dim
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
     * <tr><td> charge            <td> multiple of Electron charge (1.0 is a proton)
     * <tr><td> dipole            <td> Debye
     * <tr><td> electric field    <td> volts/Centimeter
     * <tr>
     * </table>
     */
    ELECTRON
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
    explicit units(std::string const &style);

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
     * \brief Convert
     *
     * \tparam UNIT Unit type to covert to \sa umuq::UnitType
     *
     * \param value Input value to convert to the new Style
     *
     * \note
     * \c units::convertFromStyle or \c units::convertToStyle should be called before calling this routine otherwise nothing will change
     */
    template <UnitType UNIT>
    inline void convert(double &value);

    template <UnitType UNIT>
    inline void convert(std::vector<double> &value);

    template <UnitType UNIT>
    inline void convert(double *value, int const nSize);

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

private:
    /*!
     * \brief Initialize all the dictionaries
     *
     */
    void init_dictionaries();

private:
    // Fundamental constants in SI units
    /*! \f$ Boltzmann's factor in \frac{Joule}{Kelvin} \f$ */
    double const Boltzmann = 1.38064852E-23;
    /*! \f$ Avogadro's number unitless \f$ */
    double const nAvogadro = 6.02214076E23;
    /*! \f$ The Electron rest mass in kilogram \f$ */
    double const ElectronRestMass = 9.10938356E-31;
    /*! \f$ Charge of an Electron/proton in Columb \f$ */
    double const ElectronCharge = 1.6021766208E-19;

    // Mass:
    /*! \f$ 1 atomic mass unit \f$ */
    double const Amu__T__kGram = 1.660540E-27;
    /*! \f$ 1 \frac{gram}{mol} = \frac{1E-3}{6.02214076E23} kg \f$ */
    double const Gram_mol__T__kGram = 1.660540E-27;
    // double const Gram_mol__T__kGram = 1E-3/nAvogadro;
    /*! \f$ 1 Electron mass at rest = 9.10938356E-31 kg \f$ */
    double const ElectronRestMass__T__kGram = 9.10938356E-31;
    /*! \f$ 1 gram = 1E-3 kg \f$ */
    double const Gram__T__kGram = 1E-3;
    /*! \f$ 1 pico gram = 1E-15 kg \f$ */
    double const Picogram__T__kGram = 1E-15;
    /*! \f$ 1 pico gram = 1E-15 kg \f$ */
    double const Attogram__T__kGram = 1E-21;

    // Length:
    /*! \f$ 1 Bohr = 0.529177208 Angstrom \f$ */
    double const Bohr__T__Angstrom = 0.529177208;
    /*! \f$ 1 Bohr = 5.2917721067E-11 Meter \f$ */
    double const Bohr__T__Meter = 5.2917721067E-11;
    /*! \f$ 1 Angstrom = 1E-10 Meter \f$ */
    double const Angstrom__T__Meter = 1E-10;
    /*! \f$ 1 Centimeter = 1E-2 Meter \f$ */
    double const Centimeter__T__Meter = 1E-2;
    /*! \f$ 1 Micrometer = 1E-6 Meter \f$ */
    double const Micrometer__T__Meter = 1E-6;
    /*! \f$ 1 Nanometer = 1E-9 Meter \f$ */
    double const Nanometer__T__Meter = 1E-9;

    // Time:
    /*! \f$ 1 atomic time unit = 2.418884326509E-17 second \f$ */
    double const Atu__T__Second = 2.418884326509E-17;
    /*! \f$ 1 atomic time unit = 2.418884326509E-17 second \f$ */
    double const AtuElectron__T__Second = Atu__T__Second * std::sqrt(Amu__T__kGram / ElectronRestMass);
    /*! \f$ 1 Microsecond = 1E-6 second \f$ */
    double const Microsecond__T__Second = 1E-6;
    /*! \f$ 1 Nanosecond = 1E-9 second \f$ */
    double const Nanosecond__T__Second = 1E-9;
    /*! \f$ 1 Picosecond = 1E-12 second \f$ */
    double const Picosecond__T__Second = 1E-12;
    /*! \f$ 1 Femtosecond = 1.0E-15 second \f$ */
    double const Femtosecond__T__Second = 1.0E-15;
    /*! \f$ 1 Picosecond = 1.0E03 Femtosecond \f$ */
    double const Picosecond__T__Femtosecond = 1.0E03;
    /*! \f$ 1 Femtosecond = 1.0E-03 picosecond \f$ */
    double const Femtosecond__T__Picosecond = 1.0E-03;

    // Energy:
    /*! \f$ 1 kCal = 4184.0 Joules \f$ */
    double const kCal__T__Joule = 4184.0;
    /*! \f$ 1 Hartree = 2 Rydberg = 27.211396 ElectronVolt \f$ */
    double const Hartree__T__Rydberg = 2;
    /*! \f$ 1 Hartree = 27.211396 ElectronVolt \f$ */
    double const Hartree__T__ElectronVolt = 27.211396;
    /*! \f$ 1 Hartree = 4.359744650E-18 Joules \f$ */
    double const Hartree__T__Joule = 4.359744650E-18;
    /*! \f$ 1 \frac{kJ}{mol} = 0.0103642688 \frac{ElectronVolt}{atom} \f$ */
    double const kJoule_mol__T__ElectronVolt_Atom = 0.0103642688;
    /*! \f$ 1 Joule = 6.24150965x10^(18) ElectronVolt (CODATA) \f$ */
    double const Joule__T__ElectronVolt = 6.24150965E18;
    /*! \f$ 1 ElectronVolt = 1.6021766208*10^(-19) Joules \f$ */
    double const ElectronVolt__T__Joule = 1.6021766208E-19;
    /*! \f$ 1 \frac{kCal}{mol} =  Joules \f$ */
    double const kCal_mol__T__Joule = kCal__T__Joule / nAvogadro;
    /*! \f$ 1 Erg = 1E-7 Joules \f$ */
    double const Erg__T__Joule = 1E-7;
    /*! \f$ 1 \frac{dyne}{centimeter} = 1E-7 Joules \f$ */
    double const DyneCentemeter__T__Joule = 1E-7;
    /*! \f$ 1 \frac{\text{Picogram}~\text{Micrometer}^2}{\text{Microsecond}^2} \f$ Joules \f$ */
    double const PicogramMicrometer2_Microsecond2__T__Joule = Picogram__T__kGram *
                                                              Micrometer__T__Meter *
                                                              Micrometer__T__Meter /
                                                              (Microsecond__T__Second *
                                                               Microsecond__T__Second);
    /*! \f$ {\frac{\text{Attogram}~\text{Nanometer}^2}{\text{Nanosecond}^2}} \f$ Joules */
    double const AttogramNanometer2_Nanosecond2__T__Joule = Attogram__T__kGram *
                                                            Nanometer__T__Meter *
                                                            Nanometer__T__Meter /
                                                            (Nanosecond__T__Second *
                                                             Nanosecond__T__Second);

    // Velocity:
    double const Angstrom_Femtosecond__T__Meter_Second = Angstrom__T__Meter / Femtosecond__T__Second;
    double const Angstrom_Picosecond__T__Meter_Second = Angstrom__T__Meter / Picosecond__T__Second;
    double const Micrometer_Microsecond__T__Meter_Second = Micrometer__T__Meter / Microsecond__T__Second;
    double const Nanometer_Nanosecond__T__Meter_Second = Nanometer__T__Meter / Nanosecond__T__Second;
    double const Centimeter_Second__T__Meter_Second = Centimeter__T__Meter;
    double const Bohr_AtuElectron__T__Meter_Second = Bohr__T__Meter / AtuElectron__T__Second;

    // Force:
    /*! \f$ 1 \frac{Rydberg}{Bohr} = 25.71104309541616 \frac{ElectronVolt}{Angstrom}, \f$ where Ry is Rydberg unit of energy. */
    double const Rydberg_Bohr__T__ElectronVolt_Angstrom = 25.71104309541616;
    /*! \f$ 1 \frac{Hartree}{Bohr} = 51.42208619083232 \frac{ElectronVolt}{Angstrom} \f$ */
    double const Hartree_Bohr__T__ElectronVolt_Angstrom = 51.42208619083232;

    double const kCal_moleAngstrom__T__Newton = kCal_mol__T__Joule / Angstrom__T__Meter;
    double const ElectronVolt_Angstrom__T__Newton = ElectronVolt__T__Joule / Angstrom__T__Meter;
    double const Dyne__T__Newton = DyneCentemeter__T__Joule / Centimeter__T__Meter;
    double const Hartrees_Bohr__T__Newton = DyneCentemeter__T__Joule / Centimeter__T__Meter;
    double const PicogramMicrometer_Microsecond2__T__Newton = Picogram__T__kGram *
                                                              Picogram__T__kGram /
                                                              (Microsecond__T__Second *
                                                               Microsecond__T__Second);
    double const AttogramNanometer_Nanosecond2__T__Newton = Attogram__T__kGram *
                                                            Attogram__T__kGram /
                                                            (Nanosecond__T__Second *
                                                             Nanosecond__T__Second);
    double const Hartree_Bohr__T__Newton = Hartree__T__Joule / Bohr__T__Meter;

    // Pressure:
    /*! \f$ 1 \frac{ElectronVolt}{Angstrom^3} = 160.21766208 GPa \f$ */
    double const ElectronVolt_Angstrom3__T__GigaPascal = 160.21766208;
    /*! \f$ 1 \frac{Hartree}{Bohr^3} = 29421.02648438959 GPa \f$ */
    double const Hartree_Bohr3__T__GigaPascal = 29421.02648438959;
    /*! \f$ 1 GPa = 145037.738007218 \frac{pound}{square inch} \f$ */
    double const GigaPascal__T__pound_inch2 = 145037.738007218;
    /*! \f$ 1 atm = 1.01325 bar \f$ */
    double const Atmosphere__T__Bar = 1.01325;
    /*! \f$ 1 atm =  0.000101325 GPa \f$ */
    double const Atmosphere__T__GigaPascal = 0.000101325;
    /*! \f$ 1 pascal = 1.0E-09 GPa \f$ */
    double const Pascal__T__GigaPascal = 1.0E-9;

    double const Atmosphere__T__Pascal = 101325.0;
    double const Bar__T__Pascal = 1E5;
    double const Dyne_Centimeter2__T__Pascal = DyneCentemeter__T__Joule /
                                               Centimeter__T__Meter /
                                               (Centimeter__T__Meter *
                                                Centimeter__T__Meter);
    double const Picogram_MicrometerMicrosecond2 = Picogram__T__kGram /
                                                   Micrometer__T__Meter /
                                                   (Microsecond__T__Second *
                                                    Microsecond__T__Second);
    double const Attogram_NanometerNanosecond2 = Attogram__T__kGram /
                                                 Nanometer__T__Meter /
                                                 (Nanosecond__T__Second *
                                                  Nanosecond__T__Second);

    // Viscosity
    double const Poise__T__PascalSecond = 0.1;
    double const Amu_BohrFemtosecond__T__PascalSecond = Amu__T__kGram /
                                                        (Bohr__T__Meter *
                                                         Femtosecond__T__Second);
    double const Picogram_MicrometerMicrosecond__T__PascalSecond = Picogram__T__kGram /
                                                                   (Micrometer__T__Meter *
                                                                    Microsecond__T__Second);
    double const Attogram_NanometerNanosecond__T__PascalSecond = Attogram__T__kGram /
                                                                 (Nanometer__T__Meter *
                                                                  Nanosecond__T__Second);

    // Density
    double const Gram_Centimeter3__T__kGram_Meter3 = Gram__T__kGram /
                                                     (Centimeter__T__Meter *
                                                      Centimeter__T__Meter *
                                                      Centimeter__T__Meter);
    double const Amu_Bohr3__T__kGram_Meter3 = Amu__T__kGram /
                                              (Bohr__T__Meter *
                                               Bohr__T__Meter *
                                               Bohr__T__Meter);
    double const Picogram_Micrometer3__T__kGram_Meter3 = Picogram__T__kGram /
                                                         (Micrometer__T__Meter *
                                                          Micrometer__T__Meter *
                                                          Micrometer__T__Meter);
    double const Attogram_Nanometer3__T__kGram_Meter3 = Attogram__T__kGram /
                                                        (Nanometer__T__Meter *
                                                         Nanometer__T__Meter *
                                                         Nanometer__T__Meter);

    // Charge
    double const Electron__T__Coulomb = ElectronCharge;
    double const StatCoulomb__T__Coulomb = ElectronCharge / 4.8032044E-10;
    double const PicoCoulomb__T__Coulomb = 1E-12;

    // Dipole
    double const ElectronAngstrom__T__CoulombMeter = Electron__T__Coulomb * Angstrom__T__Meter;
    double const StatCoulombCentimeter__T__CoulombMeter = StatCoulomb__T__Coulomb * Centimeter__T__Meter;
    double const Debye__T__CoulombMeter = 1E-18 * StatCoulombCentimeter__T__CoulombMeter;
    double const PicoCoulombMicrometer__T__CoulombMeter = PicoCoulomb__T__Coulomb * Micrometer__T__Meter;
    double const ElectronNanometer__T__CoulombMeter = Electron__T__Coulomb * Nanometer__T__Meter;

    // Electric field
    double const Volt_Angstrom__T__Volt_Meter = 1.0 / Angstrom__T__Meter;
    double const StatVolt_Centimeter__T__Volt_Meter = Erg__T__Joule / (StatCoulomb__T__Coulomb * Centimeter__T__Meter);
    double const Volt_Centimeter__T__Volt_Meter = 1.0 / Centimeter__T__Meter;
    double const Volt_Micrometer__T__Volt_Meter = 1.0 / Micrometer__T__Meter;
    double const Volt_Nanometer__T__Volt_Meter = 1.0 / Nanometer__T__Meter;

public:
    /*! The length scaling factor */
    double massUnitScale;
    /*! The length scaling factor */
    double lengthUnitScale;
    /*! The time scaling factor */
    double timeUnitScale;
    /*! The energy scaling factor */
    double energyUnitScale;
    /*! The velocity scaling factor */
    double velocityUnitScale;
    /*! The force scaling factor */
    double forceUnitScale;
    /*! The torque scaling factor */
    double torqueUnitScale;
    /*! The temperature scaling factor */
    double temperatureUnitScale;
    /*! The pressure scaling factor */
    double pressureUnitScale;
    /*! The viscosity scaling factor */
    double viscosityUnitScale;
    /*! The charge scaling factor */
    double chargeUnitScale;
    /*! The dipole scaling factor */
    double dipoleUnitScale;
    /*! The electricField scaling factor */
    double electricFieldUnitScale;
    /*! The density scaling factor */
    double densityUnitScale;

private:
    /*! System unit style */
    UnitStyle unitStyle;

    /*! Mass unit of the specified system style */
    MassUnit massUnit;
    /*! Length unit of the specified system style */
    LengthUnit lengthUnit;
    /*! Time unit of the specified system style */
    TimeUnit timeUnit;
    /*! Energy unit of the specified system style */
    EnergyUnit energyUnit;
    /*! Velocity unit of the specified system style */
    VelocityUnit velocityUnit;
    /*! Force unit of the specified system style */
    ForceUnit forceUnit;
    /*! Torque unit of the specified system style */
    TorqueUnit torqueUnit;
    /*! Temperature unit of the specified system style */
    TemperatureUnit temperatureUnit;
    /*! Pressure unit of the specified system style */
    PressureUnit pressureUnit;
    /*! Viscoity unit of the specified system style */
    ViscosityUnit viscosityUnit;
    /*! Charge unit of the specified system style */
    ChargeUnit chargeUnit;
    /*! Dipole unit of the specified system style */
    DipoleUnit dipoleUnit;
    /*! Electricfield unit of the specified system style */
    ElectricFieldUnit electricFieldUnit;
    /*! Density unit of the specified system style */
    DensityUnit densityUnit;

private:
    bool dictionaries_are_initialized;

    std::map<UnitStyle, MassUnit> MassUnitDic;
    std::map<UnitStyle, LengthUnit> LengthUnitDic;
    std::map<UnitStyle, TimeUnit> TimeUnitDic;
    std::map<UnitStyle, EnergyUnit> EnergyUnitDic;
    std::map<UnitStyle, VelocityUnit> VelocityUnitDic;
    std::map<UnitStyle, ForceUnit> ForceUnitDic;
    std::map<UnitStyle, TorqueUnit> TorqueUnitDic;
    std::map<UnitStyle, TemperatureUnit> TemperatureUnitDic;
    std::map<UnitStyle, PressureUnit> PressureUnitDic;
    std::map<UnitStyle, ViscosityUnit> ViscosityUnitDic;
    std::map<UnitStyle, ChargeUnit> ChargeUnitDic;
    std::map<UnitStyle, DipoleUnit> DipoleUnitDic;
    std::map<UnitStyle, ElectricFieldUnit> ElectricFieldUnitDic;
    std::map<UnitStyle, DensityUnit> DensityUnitDic;

    std::map<MassUnit, std::map<MassUnit, double>> MassUnitConvertorDic;
    std::map<LengthUnit, std::map<LengthUnit, double>> LengthUnitConvertorDic;
    std::map<TimeUnit, std::map<TimeUnit, double>> TimeUnitConvertorDic;
    std::map<EnergyUnit, std::map<EnergyUnit, double>> EnergyUnitConvertorDic;
    std::map<VelocityUnit, std::map<VelocityUnit, double>> VelocityUnitConvertorDic;
    std::map<ForceUnit, std::map<ForceUnit, double>> ForceUnitConvertorDic;
    std::map<TorqueUnit, std::map<TorqueUnit, double>> TorqueUnitConvertorDic;
    std::map<TemperatureUnit, std::map<TemperatureUnit, double>> TemperatureUnitConvertorDic;
    std::map<PressureUnit, std::map<PressureUnit, double>> PressureUnitConvertorDic;
    std::map<ViscosityUnit, std::map<ViscosityUnit, double>> ViscosityUnitConvertorDic;
    std::map<ChargeUnit, std::map<ChargeUnit, double>> ChargeUnitConvertorDic;
    std::map<DipoleUnit, std::map<DipoleUnit, double>> DipoleUnitConvertorDic;
    std::map<ElectricFieldUnit, std::map<ElectricFieldUnit, double>> ElectricFieldUnitConvertorDic;
    std::map<DensityUnit, std::map<DensityUnit, double>> DensityUnitConvertorDic;
};

units::units() : unitStyle(UnitStyle::METAL),
                 massUnit(MassUnit::Gram_Mol),
                 lengthUnit(LengthUnit::Angstrom),
                 timeUnit(TimeUnit::Picosecond),
                 energyUnit(EnergyUnit::ElectronVolt),
                 velocityUnit(VelocityUnit::Angstrom_Picosecond),
                 forceUnit(ForceUnit::ElectronVolt_Angstrom),
                 torqueUnit(TorqueUnit::ElectronVolt),
                 temperatureUnit(TemperatureUnit::Kelvin),
                 pressureUnit(PressureUnit::Bar),
                 viscosityUnit(ViscosityUnit::Poise),
                 chargeUnit(ChargeUnit::Electron),
                 dipoleUnit(DipoleUnit::ElectronAngstrom),
                 electricFieldUnit(ElectricFieldUnit::Volt_Angstrom),
                 densityUnit(DensityUnit::Gram_Centimeter3),
                 massUnitScale(1.0),
                 lengthUnitScale(1.0),
                 timeUnitScale(1.0),
                 energyUnitScale(1.0),
                 velocityUnitScale(1.0),
                 forceUnitScale(1.0),
                 torqueUnitScale(1.0),
                 temperatureUnitScale(1.0),
                 pressureUnitScale(1.0),
                 viscosityUnitScale(1.0),
                 chargeUnitScale(1.0),
                 dipoleUnitScale(1.0),
                 electricFieldUnitScale(1.0),
                 densityUnitScale(1.0),
                 dictionaries_are_initialized(false)
{
}

units::units(UnitStyle const &style)
{
    // determine unit system and set lmps_units flag
    switch (style)
    {
    case UnitStyle::REAL:
        unitStyle = UnitStyle::REAL;
        massUnit = MassUnit::Gram_Mol;
        lengthUnit = LengthUnit::Angstrom;
        timeUnit = TimeUnit::Femtosecond;
        energyUnit = EnergyUnit::kCal_mol;
        velocityUnit = VelocityUnit::Angstrom_Femtosecond;
        forceUnit = ForceUnit::kCal_moleAngstrom;
        torqueUnit = TorqueUnit::kCal_mol;
        temperatureUnit = TemperatureUnit::Kelvin;
        pressureUnit = PressureUnit::Atmosphere;
        viscosityUnit = ViscosityUnit::Poise;
        chargeUnit = ChargeUnit::Electron;
        dipoleUnit = DipoleUnit::ElectronAngstrom;
        electricFieldUnit = ElectricFieldUnit::Volt_Angstrom;
        densityUnit = DensityUnit::Gram_Centimeter3;
        break;
    case UnitStyle::METAL:
        unitStyle = UnitStyle::METAL;
        massUnit = MassUnit::Gram_Mol;
        lengthUnit = LengthUnit::Angstrom;
        timeUnit = TimeUnit::Picosecond;
        energyUnit = EnergyUnit::ElectronVolt;
        velocityUnit = VelocityUnit::Angstrom_Picosecond;
        forceUnit = ForceUnit::ElectronVolt_Angstrom;
        torqueUnit = TorqueUnit::ElectronVolt;
        temperatureUnit = TemperatureUnit::Kelvin;
        pressureUnit = PressureUnit::Bar;
        viscosityUnit = ViscosityUnit::Poise;
        chargeUnit = ChargeUnit::Electron;
        dipoleUnit = DipoleUnit::ElectronAngstrom;
        electricFieldUnit = ElectricFieldUnit::Volt_Angstrom;
        densityUnit = DensityUnit::Gram_Centimeter3;
        break;
    case UnitStyle::SI:
        unitStyle = UnitStyle::SI;
        massUnit = MassUnit::kGram;
        lengthUnit = LengthUnit::Meter;
        timeUnit = TimeUnit::Second;
        energyUnit = EnergyUnit::Joule;
        velocityUnit = VelocityUnit::Meter_Second;
        forceUnit = ForceUnit::Newton;
        torqueUnit = TorqueUnit::Newton_Meter;
        temperatureUnit = TemperatureUnit::Kelvin;
        pressureUnit = PressureUnit::Pascal;
        viscosityUnit = ViscosityUnit::PascalSecond;
        chargeUnit = ChargeUnit::Coulomb;
        dipoleUnit = DipoleUnit::CoulombMeter;
        electricFieldUnit = ElectricFieldUnit::Volt_Meter;
        densityUnit = DensityUnit::kGram_Meter3;
        break;
    case UnitStyle::CGS:
        unitStyle = UnitStyle::CGS;
        massUnit = MassUnit::Gram;
        lengthUnit = LengthUnit::Centimeter;
        timeUnit = TimeUnit::Second;
        energyUnit = EnergyUnit::Erg;
        velocityUnit = VelocityUnit::Centimeter_Second;
        forceUnit = ForceUnit::Dyne;
        torqueUnit = TorqueUnit::DyneCentimeter;
        temperatureUnit = TemperatureUnit::Kelvin;
        pressureUnit = PressureUnit::Dyne_Centimeter2;
        viscosityUnit = ViscosityUnit::Poise;
        chargeUnit = ChargeUnit::StatCoulomb;
        dipoleUnit = DipoleUnit::StatCoulombCentimeter;
        electricFieldUnit = ElectricFieldUnit::StatVolt_Centimeter;
        densityUnit = DensityUnit::Gram_Centimeter3;
        break;
    case UnitStyle::ELECTRON:
        unitStyle = UnitStyle::ELECTRON;
        massUnit = MassUnit::Amu;
        lengthUnit = LengthUnit::Bohr;
        timeUnit = TimeUnit::Femtosecond;
        energyUnit = EnergyUnit::Hartree;
        velocityUnit = VelocityUnit::Bohr_AtuElectron;
        forceUnit = ForceUnit::Hartrees_Bohr;
        torqueUnit = TorqueUnit::Hartree;
        temperatureUnit = TemperatureUnit::Kelvin;
        pressureUnit = PressureUnit::Pascal;
        viscosityUnit = ViscosityUnit::PascalSecond;
        chargeUnit = ChargeUnit::Electron;
        dipoleUnit = DipoleUnit::Debye;
        electricFieldUnit = ElectricFieldUnit::Volt_Centimeter;
        densityUnit = DensityUnit::Amu_Bohr3;
        break;
    default:
        UMUQFAIL("Unknown unit style by UMUQ!");
        break;
    }
    massUnitScale = 1.0;
    lengthUnitScale = 1.0;
    timeUnitScale = 1.0;
    energyUnitScale = 1.0;
    velocityUnitScale = 1.0;
    forceUnitScale = 1.0;
    torqueUnitScale = 1.0;
    temperatureUnitScale = 1.0;
    pressureUnitScale = 1.0;
    viscosityUnitScale = 1.0;
    chargeUnitScale = 1.0;
    dipoleUnitScale = 1.0;
    electricFieldUnitScale = 1.0;
    densityUnitScale = 1.0;
    dictionaries_are_initialized = false;
}

units::units(std::string const &style)
{
    umuq::parser p;
    auto STYLE = p.toupper(style);

    // determine unit system style and set the flags
    if (STYLE == "REAL")
    {
        unitStyle = UnitStyle::REAL;
        massUnit = MassUnit::Gram_Mol;
        lengthUnit = LengthUnit::Angstrom;
        timeUnit = TimeUnit::Femtosecond;
        energyUnit = EnergyUnit::kCal_mol;
        velocityUnit = VelocityUnit::Angstrom_Femtosecond;
        forceUnit = ForceUnit::kCal_moleAngstrom;
        torqueUnit = TorqueUnit::kCal_mol;
        temperatureUnit = TemperatureUnit::Kelvin;
        pressureUnit = PressureUnit::Atmosphere;
        viscosityUnit = ViscosityUnit::Poise;
        chargeUnit = ChargeUnit::Electron;
        dipoleUnit = DipoleUnit::ElectronAngstrom;
        electricFieldUnit = ElectricFieldUnit::Volt_Angstrom;
        densityUnit = DensityUnit::Gram_Centimeter3;
    }
    else if (STYLE == "METAL")
    {
        unitStyle = UnitStyle::METAL;
        massUnit = MassUnit::Gram_Mol;
        lengthUnit = LengthUnit::Angstrom;
        timeUnit = TimeUnit::Picosecond;
        energyUnit = EnergyUnit::ElectronVolt;
        velocityUnit = VelocityUnit::Angstrom_Picosecond;
        forceUnit = ForceUnit::ElectronVolt_Angstrom;
        torqueUnit = TorqueUnit::ElectronVolt;
        temperatureUnit = TemperatureUnit::Kelvin;
        pressureUnit = PressureUnit::Bar;
        viscosityUnit = ViscosityUnit::Poise;
        chargeUnit = ChargeUnit::Electron;
        dipoleUnit = DipoleUnit::ElectronAngstrom;
        electricFieldUnit = ElectricFieldUnit::Volt_Angstrom;
        densityUnit = DensityUnit::Gram_Centimeter3;
    }
    else if (STYLE == "SI")
    {
        unitStyle = UnitStyle::SI;
        massUnit = MassUnit::kGram;
        lengthUnit = LengthUnit::Meter;
        timeUnit = TimeUnit::Second;
        energyUnit = EnergyUnit::Joule;
        velocityUnit = VelocityUnit::Meter_Second;
        forceUnit = ForceUnit::Newton;
        torqueUnit = TorqueUnit::Newton_Meter;
        temperatureUnit = TemperatureUnit::Kelvin;
        pressureUnit = PressureUnit::Pascal;
        viscosityUnit = ViscosityUnit::PascalSecond;
        chargeUnit = ChargeUnit::Coulomb;
        dipoleUnit = DipoleUnit::CoulombMeter;
        electricFieldUnit = ElectricFieldUnit::Volt_Meter;
        densityUnit = DensityUnit::kGram_Meter3;
    }
    else if (STYLE == "CGS")
    {
        unitStyle = UnitStyle::CGS;
        massUnit = MassUnit::Gram;
        lengthUnit = LengthUnit::Centimeter;
        timeUnit = TimeUnit::Second;
        energyUnit = EnergyUnit::Erg;
        velocityUnit = VelocityUnit::Centimeter_Second;
        forceUnit = ForceUnit::Dyne;
        torqueUnit = TorqueUnit::DyneCentimeter;
        temperatureUnit = TemperatureUnit::Kelvin;
        pressureUnit = PressureUnit::Dyne_Centimeter2;
        viscosityUnit = ViscosityUnit::Poise;
        chargeUnit = ChargeUnit::StatCoulomb;
        dipoleUnit = DipoleUnit::StatCoulombCentimeter;
        electricFieldUnit = ElectricFieldUnit::StatVolt_Centimeter;
        densityUnit = DensityUnit::Gram_Centimeter3;
    }
    else if (STYLE == "ELECTRON")
    {
        unitStyle = UnitStyle::ELECTRON;
        massUnit = MassUnit::Amu;
        lengthUnit = LengthUnit::Bohr;
        timeUnit = TimeUnit::Femtosecond;
        energyUnit = EnergyUnit::Hartree;
        velocityUnit = VelocityUnit::Bohr_AtuElectron;
        forceUnit = ForceUnit::Hartrees_Bohr;
        torqueUnit = TorqueUnit::Hartree;
        temperatureUnit = TemperatureUnit::Kelvin;
        pressureUnit = PressureUnit::Pascal;
        viscosityUnit = ViscosityUnit::PascalSecond;
        chargeUnit = ChargeUnit::Electron;
        dipoleUnit = DipoleUnit::Debye;
        electricFieldUnit = ElectricFieldUnit::Volt_Centimeter;
        densityUnit = DensityUnit::Amu_Bohr3;
    }
    else
    {
        UMUQFAIL("The unit style of (", STYLE, ") is unknown by UMUQ!");
    }

    massUnitScale = 1.0;
    lengthUnitScale = 1.0;
    timeUnitScale = 1.0;
    energyUnitScale = 1.0;
    velocityUnitScale = 1.0;
    forceUnitScale = 1.0;
    torqueUnitScale = 1.0;
    temperatureUnitScale = 1.0;
    pressureUnitScale = 1.0;
    viscosityUnitScale = 1.0;
    chargeUnitScale = 1.0;
    dipoleUnitScale = 1.0;
    electricFieldUnitScale = 1.0;
    densityUnitScale = 1.0;
    dictionaries_are_initialized = false;
}

units::~units() {}

units::units(units &&other)
{
    unitStyle = std::move(other.unitStyle);
    massUnit = std::move(other.massUnit);
    lengthUnit = std::move(other.lengthUnit);
    timeUnit = std::move(other.timeUnit);
    energyUnit = std::move(other.energyUnit);
    velocityUnit = std::move(other.velocityUnit);
    forceUnit = std::move(other.forceUnit);
    torqueUnit = std::move(other.torqueUnit);
    temperatureUnit = std::move(other.temperatureUnit);
    pressureUnit = std::move(other.pressureUnit);
    viscosityUnit = std::move(other.viscosityUnit);
    chargeUnit = std::move(other.chargeUnit);
    dipoleUnit = std::move(other.dipoleUnit);
    electricFieldUnit = std::move(other.electricFieldUnit);
    densityUnit = std::move(other.densityUnit);
    massUnitScale = std::move(other.massUnitScale);
    lengthUnitScale = std::move(other.lengthUnitScale);
    timeUnitScale = std::move(other.timeUnitScale);
    energyUnitScale = std::move(other.energyUnitScale);
    velocityUnitScale = std::move(other.velocityUnitScale);
    forceUnitScale = std::move(other.forceUnitScale);
    torqueUnitScale = std::move(other.torqueUnitScale);
    temperatureUnitScale = std::move(other.temperatureUnitScale);
    pressureUnitScale = std::move(other.pressureUnitScale);
    viscosityUnitScale = std::move(other.viscosityUnitScale);
    chargeUnitScale = std::move(other.chargeUnitScale);
    dipoleUnitScale = std::move(other.dipoleUnitScale);
    electricFieldUnitScale = std::move(other.electricFieldUnitScale);
    densityUnitScale = std::move(other.densityUnitScale);
    dictionaries_are_initialized = other.dictionaries_are_initialized;
    MassUnitDic = std::move(other.MassUnitDic);
    LengthUnitDic = std::move(other.LengthUnitDic);
    TimeUnitDic = std::move(other.TimeUnitDic);
    EnergyUnitDic = std::move(other.EnergyUnitDic);
    VelocityUnitDic = std::move(other.VelocityUnitDic);
    ForceUnitDic = std::move(other.ForceUnitDic);
    TorqueUnitDic = std::move(other.TorqueUnitDic);
    TemperatureUnitDic = std::move(other.TemperatureUnitDic);
    PressureUnitDic = std::move(other.PressureUnitDic);
    ViscosityUnitDic = std::move(other.ViscosityUnitDic);
    ChargeUnitDic = std::move(other.ChargeUnitDic);
    DipoleUnitDic = std::move(other.DipoleUnitDic);
    ElectricFieldUnitDic = std::move(other.ElectricFieldUnitDic);
    DensityUnitDic = std::move(other.DensityUnitDic);
    MassUnitConvertorDic = std::move(other.MassUnitConvertorDic);
    LengthUnitConvertorDic = std::move(other.LengthUnitConvertorDic);
    TimeUnitConvertorDic = std::move(other.TimeUnitConvertorDic);
    EnergyUnitConvertorDic = std::move(other.EnergyUnitConvertorDic);
    VelocityUnitConvertorDic = std::move(other.VelocityUnitConvertorDic);
    ForceUnitConvertorDic = std::move(other.ForceUnitConvertorDic);
    TorqueUnitConvertorDic = std::move(other.TorqueUnitConvertorDic);
    TemperatureUnitConvertorDic = std::move(other.TemperatureUnitConvertorDic);
    PressureUnitConvertorDic = std::move(other.PressureUnitConvertorDic);
    ViscosityUnitConvertorDic = std::move(other.ViscosityUnitConvertorDic);
    ChargeUnitConvertorDic = std::move(other.ChargeUnitConvertorDic);
    DipoleUnitConvertorDic = std::move(other.DipoleUnitConvertorDic);
    ElectricFieldUnitConvertorDic = std::move(other.ElectricFieldUnitConvertorDic);
    DensityUnitConvertorDic = std::move(other.DensityUnitConvertorDic);
}

units &units::operator=(units &&other)
{
    unitStyle = std::move(other.unitStyle);
    massUnit = std::move(other.massUnit);
    lengthUnit = std::move(other.lengthUnit);
    timeUnit = std::move(other.timeUnit);
    energyUnit = std::move(other.energyUnit);
    velocityUnit = std::move(other.velocityUnit);
    forceUnit = std::move(other.forceUnit);
    torqueUnit = std::move(other.torqueUnit);
    temperatureUnit = std::move(other.temperatureUnit);
    pressureUnit = std::move(other.pressureUnit);
    viscosityUnit = std::move(other.viscosityUnit);
    chargeUnit = std::move(other.chargeUnit);
    dipoleUnit = std::move(other.dipoleUnit);
    electricFieldUnit = std::move(other.electricFieldUnit);
    densityUnit = std::move(other.densityUnit);
    massUnitScale = std::move(other.massUnitScale);
    lengthUnitScale = std::move(other.lengthUnitScale);
    timeUnitScale = std::move(other.timeUnitScale);
    energyUnitScale = std::move(other.energyUnitScale);
    velocityUnitScale = std::move(other.velocityUnitScale);
    forceUnitScale = std::move(other.forceUnitScale);
    torqueUnitScale = std::move(other.torqueUnitScale);
    temperatureUnitScale = std::move(other.temperatureUnitScale);
    pressureUnitScale = std::move(other.pressureUnitScale);
    viscosityUnitScale = std::move(other.viscosityUnitScale);
    chargeUnitScale = std::move(other.chargeUnitScale);
    dipoleUnitScale = std::move(other.dipoleUnitScale);
    electricFieldUnitScale = std::move(other.electricFieldUnitScale);
    densityUnitScale = std::move(other.densityUnitScale);
    dictionaries_are_initialized = other.dictionaries_are_initialized;
    MassUnitDic = std::move(other.MassUnitDic);
    LengthUnitDic = std::move(other.LengthUnitDic);
    TimeUnitDic = std::move(other.TimeUnitDic);
    EnergyUnitDic = std::move(other.EnergyUnitDic);
    VelocityUnitDic = std::move(other.VelocityUnitDic);
    ForceUnitDic = std::move(other.ForceUnitDic);
    TorqueUnitDic = std::move(other.TorqueUnitDic);
    TemperatureUnitDic = std::move(other.TemperatureUnitDic);
    PressureUnitDic = std::move(other.PressureUnitDic);
    ViscosityUnitDic = std::move(other.ViscosityUnitDic);
    ChargeUnitDic = std::move(other.ChargeUnitDic);
    DipoleUnitDic = std::move(other.DipoleUnitDic);
    ElectricFieldUnitDic = std::move(other.ElectricFieldUnitDic);
    DensityUnitDic = std::move(other.DensityUnitDic);
    MassUnitConvertorDic = std::move(other.MassUnitConvertorDic);
    LengthUnitConvertorDic = std::move(other.LengthUnitConvertorDic);
    TimeUnitConvertorDic = std::move(other.TimeUnitConvertorDic);
    EnergyUnitConvertorDic = std::move(other.EnergyUnitConvertorDic);
    VelocityUnitConvertorDic = std::move(other.VelocityUnitConvertorDic);
    ForceUnitConvertorDic = std::move(other.ForceUnitConvertorDic);
    TorqueUnitConvertorDic = std::move(other.TorqueUnitConvertorDic);
    TemperatureUnitConvertorDic = std::move(other.TemperatureUnitConvertorDic);
    PressureUnitConvertorDic = std::move(other.PressureUnitConvertorDic);
    ViscosityUnitConvertorDic = std::move(other.ViscosityUnitConvertorDic);
    ChargeUnitConvertorDic = std::move(other.ChargeUnitConvertorDic);
    DipoleUnitConvertorDic = std::move(other.DipoleUnitConvertorDic);
    ElectricFieldUnitConvertorDic = std::move(other.ElectricFieldUnitConvertorDic);
    DensityUnitConvertorDic = std::move(other.DensityUnitConvertorDic);

    return *this;
}

void units::init_dictionaries()
{
    if (dictionaries_are_initialized)
    {
        return;
    }

    MassUnitDic[UnitStyle::REAL] = MassUnit::Gram_Mol;
    MassUnitDic[UnitStyle::METAL] = MassUnit::Gram_Mol;
    MassUnitDic[UnitStyle::SI] = MassUnit::kGram;
    MassUnitDic[UnitStyle::CGS] = MassUnit::Gram;
    MassUnitDic[UnitStyle::ELECTRON] = MassUnit::Amu;

    LengthUnitDic[UnitStyle::REAL] = LengthUnit::Angstrom;
    LengthUnitDic[UnitStyle::METAL] = LengthUnit::Angstrom;
    LengthUnitDic[UnitStyle::SI] = LengthUnit::Meter;
    LengthUnitDic[UnitStyle::CGS] = LengthUnit::Centimeter;
    LengthUnitDic[UnitStyle::ELECTRON] = LengthUnit::Bohr;

    TimeUnitDic[UnitStyle::REAL] = TimeUnit::Femtosecond;
    TimeUnitDic[UnitStyle::METAL] = TimeUnit::Picosecond;
    TimeUnitDic[UnitStyle::SI] = TimeUnit::Second;
    TimeUnitDic[UnitStyle::CGS] = TimeUnit::Second;
    TimeUnitDic[UnitStyle::ELECTRON] = TimeUnit::Femtosecond;

    EnergyUnitDic[UnitStyle::REAL] = EnergyUnit::kCal_mol;
    EnergyUnitDic[UnitStyle::METAL] = EnergyUnit::ElectronVolt;
    EnergyUnitDic[UnitStyle::SI] = EnergyUnit::Joule;
    EnergyUnitDic[UnitStyle::CGS] = EnergyUnit::Erg;
    EnergyUnitDic[UnitStyle::ELECTRON] = EnergyUnit::Hartree;

    VelocityUnitDic[UnitStyle::REAL] = VelocityUnit::Angstrom_Femtosecond;
    VelocityUnitDic[UnitStyle::METAL] = VelocityUnit::Angstrom_Picosecond;
    VelocityUnitDic[UnitStyle::SI] = VelocityUnit::Meter_Second;
    VelocityUnitDic[UnitStyle::CGS] = VelocityUnit::Centimeter_Second;
    VelocityUnitDic[UnitStyle::ELECTRON] = VelocityUnit::Bohr_AtuElectron;

    ForceUnitDic[UnitStyle::REAL] = ForceUnit::kCal_moleAngstrom;
    ForceUnitDic[UnitStyle::METAL] = ForceUnit::ElectronVolt_Angstrom;
    ForceUnitDic[UnitStyle::SI] = ForceUnit::Newton;
    ForceUnitDic[UnitStyle::CGS] = ForceUnit::Dyne;
    ForceUnitDic[UnitStyle::ELECTRON] = ForceUnit::Hartrees_Bohr;

    TorqueUnitDic[UnitStyle::REAL] = TorqueUnit::kCal_mol;
    TorqueUnitDic[UnitStyle::METAL] = TorqueUnit::ElectronVolt;
    TorqueUnitDic[UnitStyle::SI] = TorqueUnit::Newton_Meter;
    TorqueUnitDic[UnitStyle::CGS] = TorqueUnit::DyneCentimeter;
    TorqueUnitDic[UnitStyle::ELECTRON] = TorqueUnit::Hartree;

    TemperatureUnitDic[UnitStyle::REAL] = TemperatureUnit::Kelvin;
    TemperatureUnitDic[UnitStyle::METAL] = TemperatureUnit::Kelvin;
    TemperatureUnitDic[UnitStyle::SI] = TemperatureUnit::Kelvin;
    TemperatureUnitDic[UnitStyle::CGS] = TemperatureUnit::Kelvin;
    TemperatureUnitDic[UnitStyle::ELECTRON] = TemperatureUnit::Kelvin;

    PressureUnitDic[UnitStyle::REAL] = PressureUnit::Atmosphere;
    PressureUnitDic[UnitStyle::METAL] = PressureUnit::Bar;
    PressureUnitDic[UnitStyle::SI] = PressureUnit::Pascal;
    PressureUnitDic[UnitStyle::CGS] = PressureUnit::Dyne_Centimeter2;
    PressureUnitDic[UnitStyle::ELECTRON] = PressureUnit::Pascal;

    ViscosityUnitDic[UnitStyle::REAL] = ViscosityUnit::Poise;
    ViscosityUnitDic[UnitStyle::METAL] = ViscosityUnit::Poise;
    ViscosityUnitDic[UnitStyle::SI] = ViscosityUnit::PascalSecond;
    ViscosityUnitDic[UnitStyle::CGS] = ViscosityUnit::Poise;
    ViscosityUnitDic[UnitStyle::ELECTRON] = ViscosityUnit::PascalSecond;

    ChargeUnitDic[UnitStyle::REAL] = ChargeUnit::Electron;
    ChargeUnitDic[UnitStyle::METAL] = ChargeUnit::Electron;
    ChargeUnitDic[UnitStyle::SI] = ChargeUnit::Coulomb;
    ChargeUnitDic[UnitStyle::CGS] = ChargeUnit::StatCoulomb;
    ChargeUnitDic[UnitStyle::ELECTRON] = ChargeUnit::Electron;

    DipoleUnitDic[UnitStyle::REAL] = DipoleUnit::ElectronAngstrom;
    DipoleUnitDic[UnitStyle::METAL] = DipoleUnit::ElectronAngstrom;
    DipoleUnitDic[UnitStyle::SI] = DipoleUnit::CoulombMeter;
    DipoleUnitDic[UnitStyle::CGS] = DipoleUnit::StatCoulombCentimeter;
    DipoleUnitDic[UnitStyle::ELECTRON] = DipoleUnit::Debye;

    ElectricFieldUnitDic[UnitStyle::REAL] = ElectricFieldUnit::Volt_Angstrom;
    ElectricFieldUnitDic[UnitStyle::METAL] = ElectricFieldUnit::Volt_Angstrom;
    ElectricFieldUnitDic[UnitStyle::SI] = ElectricFieldUnit::Volt_Meter;
    ElectricFieldUnitDic[UnitStyle::CGS] = ElectricFieldUnit::StatVolt_Centimeter;
    ElectricFieldUnitDic[UnitStyle::ELECTRON] = ElectricFieldUnit::Volt_Centimeter;

    DensityUnitDic[UnitStyle::REAL] = DensityUnit::Gram_Centimeter3;
    DensityUnitDic[UnitStyle::METAL] = DensityUnit::Gram_Centimeter3;
    DensityUnitDic[UnitStyle::SI] = DensityUnit::kGram_Meter3;
    DensityUnitDic[UnitStyle::CGS] = DensityUnit::Gram_Centimeter3;
    DensityUnitDic[UnitStyle::ELECTRON] = DensityUnit::Amu_Bohr3;

    MassUnitConvertorDic[MassUnit::kGram][MassUnit::kGram] = 1.0;
    MassUnitConvertorDic[MassUnit::kGram][MassUnit::Gram_Mol] = 1.0 / Gram_mol__T__kGram;
    MassUnitConvertorDic[MassUnit::kGram][MassUnit::Gram] = 1.0 / Gram__T__kGram;
    MassUnitConvertorDic[MassUnit::kGram][MassUnit::Amu] = 1.0 / Amu__T__kGram;

    MassUnitConvertorDic[MassUnit::Gram_Mol][MassUnit::kGram] = Gram_mol__T__kGram;
    MassUnitConvertorDic[MassUnit::Gram_Mol][MassUnit::Gram_Mol] = 1.0;
    MassUnitConvertorDic[MassUnit::Gram_Mol][MassUnit::Gram] = Gram_mol__T__kGram * 1.0 / Gram__T__kGram;
    MassUnitConvertorDic[MassUnit::Gram_Mol][MassUnit::Amu] = Gram_mol__T__kGram * 1.0 / Amu__T__kGram;

    MassUnitConvertorDic[MassUnit::Gram][MassUnit::kGram] = Gram__T__kGram;
    MassUnitConvertorDic[MassUnit::Gram][MassUnit::Gram_Mol] = Gram__T__kGram * 1.0 / Gram_mol__T__kGram;
    MassUnitConvertorDic[MassUnit::Gram][MassUnit::Gram] = 1.0;
    MassUnitConvertorDic[MassUnit::Gram][MassUnit::Amu] = Gram__T__kGram * 1.0 / Amu__T__kGram;

    MassUnitConvertorDic[MassUnit::Amu][MassUnit::kGram] = Amu__T__kGram;
    MassUnitConvertorDic[MassUnit::Amu][MassUnit::Gram_Mol] = Amu__T__kGram * 1.0 / Gram_mol__T__kGram;
    MassUnitConvertorDic[MassUnit::Amu][MassUnit::Gram] = Amu__T__kGram * 1.0 / Gram__T__kGram;
    MassUnitConvertorDic[MassUnit::Amu][MassUnit::Amu] = 1.0;

    LengthUnitConvertorDic[LengthUnit::Meter][LengthUnit::Meter] = 1.0;
    LengthUnitConvertorDic[LengthUnit::Meter][LengthUnit::Angstrom] = 1.0 / Angstrom__T__Meter;
    LengthUnitConvertorDic[LengthUnit::Meter][LengthUnit::Centimeter] = 1.0 / Centimeter__T__Meter;
    LengthUnitConvertorDic[LengthUnit::Meter][LengthUnit::Bohr] = 1.0 / Bohr__T__Meter;

    LengthUnitConvertorDic[LengthUnit::Angstrom][LengthUnit::Meter] = Angstrom__T__Meter;
    LengthUnitConvertorDic[LengthUnit::Angstrom][LengthUnit::Angstrom] = 1.0;
    LengthUnitConvertorDic[LengthUnit::Angstrom][LengthUnit::Centimeter] = Angstrom__T__Meter * 1.0 / Centimeter__T__Meter;
    LengthUnitConvertorDic[LengthUnit::Angstrom][LengthUnit::Bohr] = 1.0 / Bohr__T__Angstrom;

    LengthUnitConvertorDic[LengthUnit::Centimeter][LengthUnit::Meter] = Centimeter__T__Meter;
    LengthUnitConvertorDic[LengthUnit::Centimeter][LengthUnit::Angstrom] = Centimeter__T__Meter * 1.0 / Angstrom__T__Meter;
    LengthUnitConvertorDic[LengthUnit::Centimeter][LengthUnit::Centimeter] = 1.0;
    LengthUnitConvertorDic[LengthUnit::Centimeter][LengthUnit::Bohr] = Centimeter__T__Meter * 1.0 / Bohr__T__Meter;

    LengthUnitConvertorDic[LengthUnit::Bohr][LengthUnit::Meter] = Bohr__T__Meter;
    LengthUnitConvertorDic[LengthUnit::Bohr][LengthUnit::Angstrom] = Bohr__T__Angstrom;
    LengthUnitConvertorDic[LengthUnit::Bohr][LengthUnit::Centimeter] = Bohr__T__Meter * 1.0 / Centimeter__T__Meter;
    LengthUnitConvertorDic[LengthUnit::Bohr][LengthUnit::Bohr] = 1.0;

    TimeUnitConvertorDic[TimeUnit::Second][TimeUnit::Second] = 1.0;
    TimeUnitConvertorDic[TimeUnit::Second][TimeUnit::Femtosecond] = 1.0 / Femtosecond__T__Second;
    TimeUnitConvertorDic[TimeUnit::Second][TimeUnit::Picosecond] = 1.0 / Picosecond__T__Second;

    TimeUnitConvertorDic[TimeUnit::Femtosecond][TimeUnit::Second] = Femtosecond__T__Second;
    TimeUnitConvertorDic[TimeUnit::Femtosecond][TimeUnit::Femtosecond] = 1.0;
    TimeUnitConvertorDic[TimeUnit::Femtosecond][TimeUnit::Picosecond] = Femtosecond__T__Picosecond;

    TimeUnitConvertorDic[TimeUnit::Picosecond][TimeUnit::Second] = Picosecond__T__Second;
    TimeUnitConvertorDic[TimeUnit::Picosecond][TimeUnit::Femtosecond] = Picosecond__T__Femtosecond;
    TimeUnitConvertorDic[TimeUnit::Picosecond][TimeUnit::Picosecond] = 1.0;

    EnergyUnitConvertorDic[EnergyUnit::Joule][EnergyUnit::Joule] = 1.0;
    EnergyUnitConvertorDic[EnergyUnit::Joule][EnergyUnit::kCal_mol] = 1.0 / kCal__T__Joule;
    EnergyUnitConvertorDic[EnergyUnit::Joule][EnergyUnit::ElectronVolt] = Joule__T__ElectronVolt;
    EnergyUnitConvertorDic[EnergyUnit::Joule][EnergyUnit::Erg] = 1.0 / Erg__T__Joule;
    EnergyUnitConvertorDic[EnergyUnit::Joule][EnergyUnit::Hartree] = 1.0 / Hartree__T__Joule;

    EnergyUnitConvertorDic[EnergyUnit::kCal_mol][EnergyUnit::Joule] = kCal__T__Joule;
    EnergyUnitConvertorDic[EnergyUnit::kCal_mol][EnergyUnit::kCal_mol] = 1.0;
    EnergyUnitConvertorDic[EnergyUnit::kCal_mol][EnergyUnit::ElectronVolt] = kCal__T__Joule * Joule__T__ElectronVolt;
    EnergyUnitConvertorDic[EnergyUnit::kCal_mol][EnergyUnit::Erg] = kCal__T__Joule * 1.0 / Erg__T__Joule;
    EnergyUnitConvertorDic[EnergyUnit::kCal_mol][EnergyUnit::Hartree] = kCal__T__Joule * 1.0 / Hartree__T__Joule;

    EnergyUnitConvertorDic[EnergyUnit::ElectronVolt][EnergyUnit::Joule] = ElectronVolt__T__Joule;
    EnergyUnitConvertorDic[EnergyUnit::ElectronVolt][EnergyUnit::kCal_mol] = ElectronVolt__T__Joule * 1.0 / kCal_mol__T__Joule;
    EnergyUnitConvertorDic[EnergyUnit::ElectronVolt][EnergyUnit::ElectronVolt] = 1.0;
    EnergyUnitConvertorDic[EnergyUnit::ElectronVolt][EnergyUnit::Erg] = ElectronVolt__T__Joule * 1.0 / Erg__T__Joule;
    EnergyUnitConvertorDic[EnergyUnit::ElectronVolt][EnergyUnit::Hartree] = 1.0 / Hartree__T__ElectronVolt;

    EnergyUnitConvertorDic[EnergyUnit::Erg][EnergyUnit::Joule] = Erg__T__Joule;
    EnergyUnitConvertorDic[EnergyUnit::Erg][EnergyUnit::kCal_mol] = Erg__T__Joule * 1.0 / kCal_mol__T__Joule;
    EnergyUnitConvertorDic[EnergyUnit::Erg][EnergyUnit::ElectronVolt] = Erg__T__Joule * 1.0 / ElectronVolt__T__Joule;
    EnergyUnitConvertorDic[EnergyUnit::Erg][EnergyUnit::Erg] = 1.0;
    EnergyUnitConvertorDic[EnergyUnit::Erg][EnergyUnit::Hartree] = Erg__T__Joule * 1.0 / Hartree__T__Joule;

    EnergyUnitConvertorDic[EnergyUnit::Hartree][EnergyUnit::Joule] = Hartree__T__Joule;
    EnergyUnitConvertorDic[EnergyUnit::Hartree][EnergyUnit::kCal_mol] = Hartree__T__Joule * 1.0 / kCal_mol__T__Joule;
    EnergyUnitConvertorDic[EnergyUnit::Hartree][EnergyUnit::ElectronVolt] = Hartree__T__ElectronVolt;
    EnergyUnitConvertorDic[EnergyUnit::Hartree][EnergyUnit::Erg] = Hartree__T__Joule * 1.0 / Erg__T__Joule;
    EnergyUnitConvertorDic[EnergyUnit::Hartree][EnergyUnit::Hartree] = 1.0;

    VelocityUnitConvertorDic[VelocityUnit::Meter_Second][VelocityUnit::Meter_Second] = 1.0;
    VelocityUnitConvertorDic[VelocityUnit::Meter_Second][VelocityUnit::Angstrom_Femtosecond] = 1.0 / Angstrom_Femtosecond__T__Meter_Second;
    VelocityUnitConvertorDic[VelocityUnit::Meter_Second][VelocityUnit::Angstrom_Picosecond] = 1.0 / Angstrom_Picosecond__T__Meter_Second;
    VelocityUnitConvertorDic[VelocityUnit::Meter_Second][VelocityUnit::Centimeter_Second] = 1.0 / Centimeter_Second__T__Meter_Second;
    VelocityUnitConvertorDic[VelocityUnit::Meter_Second][VelocityUnit::Bohr_AtuElectron] = 1.0 / Bohr_AtuElectron__T__Meter_Second;

    VelocityUnitConvertorDic[VelocityUnit::Angstrom_Femtosecond][VelocityUnit::Meter_Second] = Angstrom_Femtosecond__T__Meter_Second;
    VelocityUnitConvertorDic[VelocityUnit::Angstrom_Femtosecond][VelocityUnit::Angstrom_Femtosecond] = 1.0;
    VelocityUnitConvertorDic[VelocityUnit::Angstrom_Femtosecond][VelocityUnit::Angstrom_Picosecond] = Angstrom_Femtosecond__T__Meter_Second * 1.0 / Angstrom_Picosecond__T__Meter_Second;
    VelocityUnitConvertorDic[VelocityUnit::Angstrom_Femtosecond][VelocityUnit::Centimeter_Second] = Angstrom_Femtosecond__T__Meter_Second * 1.0 / Centimeter_Second__T__Meter_Second;
    VelocityUnitConvertorDic[VelocityUnit::Angstrom_Femtosecond][VelocityUnit::Bohr_AtuElectron] = Angstrom_Femtosecond__T__Meter_Second * 1.0 / Bohr_AtuElectron__T__Meter_Second;

    VelocityUnitConvertorDic[VelocityUnit::Angstrom_Picosecond][VelocityUnit::Meter_Second] = Angstrom_Picosecond__T__Meter_Second;
    VelocityUnitConvertorDic[VelocityUnit::Angstrom_Picosecond][VelocityUnit::Angstrom_Femtosecond] = Angstrom_Picosecond__T__Meter_Second * 1.0 / Angstrom_Femtosecond__T__Meter_Second;
    VelocityUnitConvertorDic[VelocityUnit::Angstrom_Picosecond][VelocityUnit::Angstrom_Picosecond] = 1.0;
    VelocityUnitConvertorDic[VelocityUnit::Angstrom_Picosecond][VelocityUnit::Centimeter_Second] = Angstrom_Picosecond__T__Meter_Second * 1.0 / Centimeter_Second__T__Meter_Second;
    VelocityUnitConvertorDic[VelocityUnit::Angstrom_Picosecond][VelocityUnit::Bohr_AtuElectron] = Angstrom_Picosecond__T__Meter_Second * 1.0 / Bohr_AtuElectron__T__Meter_Second;

    VelocityUnitConvertorDic[VelocityUnit::Centimeter_Second][VelocityUnit::Meter_Second] = Centimeter_Second__T__Meter_Second;
    VelocityUnitConvertorDic[VelocityUnit::Centimeter_Second][VelocityUnit::Angstrom_Femtosecond] = Centimeter_Second__T__Meter_Second * 1.0 / Angstrom_Femtosecond__T__Meter_Second;
    VelocityUnitConvertorDic[VelocityUnit::Centimeter_Second][VelocityUnit::Angstrom_Picosecond] = Centimeter_Second__T__Meter_Second * 1.0 / Angstrom_Picosecond__T__Meter_Second;
    VelocityUnitConvertorDic[VelocityUnit::Centimeter_Second][VelocityUnit::Centimeter_Second] = 1.0;
    VelocityUnitConvertorDic[VelocityUnit::Centimeter_Second][VelocityUnit::Bohr_AtuElectron] = Centimeter_Second__T__Meter_Second * 1.0 / Bohr_AtuElectron__T__Meter_Second;

    VelocityUnitConvertorDic[VelocityUnit::Bohr_AtuElectron][VelocityUnit::Meter_Second] = Bohr_AtuElectron__T__Meter_Second;
    VelocityUnitConvertorDic[VelocityUnit::Bohr_AtuElectron][VelocityUnit::Angstrom_Femtosecond] = Bohr_AtuElectron__T__Meter_Second * 1.0 / Angstrom_Femtosecond__T__Meter_Second;
    VelocityUnitConvertorDic[VelocityUnit::Bohr_AtuElectron][VelocityUnit::Angstrom_Picosecond] = Bohr_AtuElectron__T__Meter_Second * 1.0 / Angstrom_Picosecond__T__Meter_Second;
    VelocityUnitConvertorDic[VelocityUnit::Bohr_AtuElectron][VelocityUnit::Centimeter_Second] = Bohr_AtuElectron__T__Meter_Second * 1.0 / Centimeter_Second__T__Meter_Second;
    VelocityUnitConvertorDic[VelocityUnit::Bohr_AtuElectron][VelocityUnit::Bohr_AtuElectron] = 1.0;

    ForceUnitConvertorDic[ForceUnit::Newton][ForceUnit::Newton] = 1.0;
    ForceUnitConvertorDic[ForceUnit::Newton][ForceUnit::kCal_moleAngstrom] = 1.0 / kCal_moleAngstrom__T__Newton;
    ForceUnitConvertorDic[ForceUnit::Newton][ForceUnit::ElectronVolt_Angstrom] = 1.0 / ElectronVolt_Angstrom__T__Newton;
    ForceUnitConvertorDic[ForceUnit::Newton][ForceUnit::Dyne] = 1.0 / Dyne__T__Newton;
    ForceUnitConvertorDic[ForceUnit::Newton][ForceUnit::Hartrees_Bohr] = 1.0 / Hartree_Bohr__T__Newton;

    ForceUnitConvertorDic[ForceUnit::kCal_moleAngstrom][ForceUnit::Newton] = kCal_moleAngstrom__T__Newton;
    ForceUnitConvertorDic[ForceUnit::kCal_moleAngstrom][ForceUnit::kCal_moleAngstrom] = 1.0;
    ForceUnitConvertorDic[ForceUnit::kCal_moleAngstrom][ForceUnit::ElectronVolt_Angstrom] = kCal_moleAngstrom__T__Newton * 1.0 / ElectronVolt_Angstrom__T__Newton;
    ForceUnitConvertorDic[ForceUnit::kCal_moleAngstrom][ForceUnit::Dyne] = kCal_moleAngstrom__T__Newton * 1.0 / Dyne__T__Newton;
    ForceUnitConvertorDic[ForceUnit::kCal_moleAngstrom][ForceUnit::Hartrees_Bohr] = kCal_moleAngstrom__T__Newton * 1.0 / Hartree_Bohr__T__Newton;

    ForceUnitConvertorDic[ForceUnit::ElectronVolt_Angstrom][ForceUnit::Newton] = ElectronVolt_Angstrom__T__Newton;
    ForceUnitConvertorDic[ForceUnit::ElectronVolt_Angstrom][ForceUnit::kCal_moleAngstrom] = ElectronVolt_Angstrom__T__Newton * 1.0 / kCal_moleAngstrom__T__Newton;
    ForceUnitConvertorDic[ForceUnit::ElectronVolt_Angstrom][ForceUnit::ElectronVolt_Angstrom] = 1.0;
    ForceUnitConvertorDic[ForceUnit::ElectronVolt_Angstrom][ForceUnit::Dyne] = ElectronVolt_Angstrom__T__Newton * 1.0 / Dyne__T__Newton;
    ForceUnitConvertorDic[ForceUnit::ElectronVolt_Angstrom][ForceUnit::Hartrees_Bohr] = ElectronVolt_Angstrom__T__Newton * 1.0 / Hartree_Bohr__T__Newton;

    ForceUnitConvertorDic[ForceUnit::Dyne][ForceUnit::Newton] = Dyne__T__Newton;
    ForceUnitConvertorDic[ForceUnit::Dyne][ForceUnit::kCal_moleAngstrom] = Dyne__T__Newton * 1.0 / kCal_moleAngstrom__T__Newton;
    ForceUnitConvertorDic[ForceUnit::Dyne][ForceUnit::ElectronVolt_Angstrom] = Dyne__T__Newton * 1.0 / ElectronVolt_Angstrom__T__Newton;
    ForceUnitConvertorDic[ForceUnit::Dyne][ForceUnit::Dyne] = 1.0;
    ForceUnitConvertorDic[ForceUnit::Dyne][ForceUnit::Hartrees_Bohr] = Dyne__T__Newton * 1.0 / Hartree_Bohr__T__Newton;

    ForceUnitConvertorDic[ForceUnit::Hartrees_Bohr][ForceUnit::Newton] = Hartree_Bohr__T__Newton;
    ForceUnitConvertorDic[ForceUnit::Hartrees_Bohr][ForceUnit::kCal_moleAngstrom] = Hartree_Bohr__T__Newton * 1.0 / kCal_moleAngstrom__T__Newton;
    ForceUnitConvertorDic[ForceUnit::Hartrees_Bohr][ForceUnit::ElectronVolt_Angstrom] = Hartree_Bohr__T__ElectronVolt_Angstrom;
    ForceUnitConvertorDic[ForceUnit::Hartrees_Bohr][ForceUnit::Dyne] = Hartree_Bohr__T__Newton * 1.0 / Dyne__T__Newton;
    ForceUnitConvertorDic[ForceUnit::Hartrees_Bohr][ForceUnit::Hartrees_Bohr] = 1.0;

    TorqueUnitConvertorDic[TorqueUnit::Newton_Meter][TorqueUnit::Newton_Meter] = 1.0;
    TorqueUnitConvertorDic[TorqueUnit::Newton_Meter][TorqueUnit::kCal_mol] = 1.0 / kCal_mol__T__Joule;
    TorqueUnitConvertorDic[TorqueUnit::Newton_Meter][TorqueUnit::ElectronVolt] = Joule__T__ElectronVolt;
    TorqueUnitConvertorDic[TorqueUnit::Newton_Meter][TorqueUnit::DyneCentimeter] = 1.0 / DyneCentemeter__T__Joule;
    TorqueUnitConvertorDic[TorqueUnit::Newton_Meter][TorqueUnit::Hartree] = 1.0 / Hartree__T__Joule;

    TorqueUnitConvertorDic[TorqueUnit::kCal_mol][TorqueUnit::Newton_Meter] = kCal_mol__T__Joule;
    TorqueUnitConvertorDic[TorqueUnit::kCal_mol][TorqueUnit::kCal_mol] = 1.0;
    TorqueUnitConvertorDic[TorqueUnit::kCal_mol][TorqueUnit::ElectronVolt] = kCal_mol__T__Joule * Joule__T__ElectronVolt;
    TorqueUnitConvertorDic[TorqueUnit::kCal_mol][TorqueUnit::DyneCentimeter] = kCal_mol__T__Joule * 1.0 / DyneCentemeter__T__Joule;
    TorqueUnitConvertorDic[TorqueUnit::kCal_mol][TorqueUnit::Hartree] = kCal_mol__T__Joule * 1.0 / Hartree__T__Joule;

    TorqueUnitConvertorDic[TorqueUnit::ElectronVolt][TorqueUnit::Newton_Meter] = ElectronVolt__T__Joule;
    TorqueUnitConvertorDic[TorqueUnit::ElectronVolt][TorqueUnit::kCal_mol] = ElectronVolt__T__Joule * 1.0 / kCal_mol__T__Joule;
    TorqueUnitConvertorDic[TorqueUnit::ElectronVolt][TorqueUnit::ElectronVolt] = 1.0;
    TorqueUnitConvertorDic[TorqueUnit::ElectronVolt][TorqueUnit::DyneCentimeter] = ElectronVolt__T__Joule * 1.0 / DyneCentemeter__T__Joule;
    TorqueUnitConvertorDic[TorqueUnit::ElectronVolt][TorqueUnit::Hartree] = ElectronVolt__T__Joule * 1.0 / Hartree__T__Joule;

    TorqueUnitConvertorDic[TorqueUnit::DyneCentimeter][TorqueUnit::Newton_Meter] = DyneCentemeter__T__Joule;
    TorqueUnitConvertorDic[TorqueUnit::DyneCentimeter][TorqueUnit::kCal_mol] = DyneCentemeter__T__Joule * 1.0 / kCal_mol__T__Joule;
    TorqueUnitConvertorDic[TorqueUnit::DyneCentimeter][TorqueUnit::ElectronVolt] = DyneCentemeter__T__Joule * Joule__T__ElectronVolt;
    TorqueUnitConvertorDic[TorqueUnit::DyneCentimeter][TorqueUnit::DyneCentimeter] = 1.0;
    TorqueUnitConvertorDic[TorqueUnit::DyneCentimeter][TorqueUnit::Hartree] = DyneCentemeter__T__Joule * 1.0 / Hartree__T__Joule;

    TorqueUnitConvertorDic[TorqueUnit::Hartree][TorqueUnit::Newton_Meter] = Hartree__T__Joule;
    TorqueUnitConvertorDic[TorqueUnit::Hartree][TorqueUnit::kCal_mol] = Hartree__T__Joule * 1.0 / kCal_mol__T__Joule;
    TorqueUnitConvertorDic[TorqueUnit::Hartree][TorqueUnit::ElectronVolt] = Hartree__T__ElectronVolt;
    TorqueUnitConvertorDic[TorqueUnit::Hartree][TorqueUnit::DyneCentimeter] = Hartree__T__Joule * 1.0 / DyneCentemeter__T__Joule;
    TorqueUnitConvertorDic[TorqueUnit::Hartree][TorqueUnit::Hartree] = 1.0;

    TemperatureUnitConvertorDic[TemperatureUnit::Kelvin][TemperatureUnit::Kelvin] = 1.0;

    PressureUnitConvertorDic[PressureUnit::Pascal][PressureUnit::Pascal] = 1.0;
    PressureUnitConvertorDic[PressureUnit::Pascal][PressureUnit::Atmosphere] = 1.0 / Atmosphere__T__Pascal;
    PressureUnitConvertorDic[PressureUnit::Pascal][PressureUnit::Bar] = 1.0 / Bar__T__Pascal;
    PressureUnitConvertorDic[PressureUnit::Pascal][PressureUnit::Dyne_Centimeter2] = 1.0 / Dyne_Centimeter2__T__Pascal;

    PressureUnitConvertorDic[PressureUnit::Atmosphere][PressureUnit::Pascal] = Atmosphere__T__Pascal;
    PressureUnitConvertorDic[PressureUnit::Atmosphere][PressureUnit::Atmosphere] = 1.0;
    PressureUnitConvertorDic[PressureUnit::Atmosphere][PressureUnit::Bar] = Atmosphere__T__Bar;
    PressureUnitConvertorDic[PressureUnit::Atmosphere][PressureUnit::Dyne_Centimeter2] = Atmosphere__T__Pascal * 1.0 / Dyne_Centimeter2__T__Pascal;

    PressureUnitConvertorDic[PressureUnit::Bar][PressureUnit::Pascal] = Bar__T__Pascal;
    PressureUnitConvertorDic[PressureUnit::Bar][PressureUnit::Atmosphere] = Bar__T__Pascal * 1.0 / Atmosphere__T__Pascal;
    PressureUnitConvertorDic[PressureUnit::Bar][PressureUnit::Bar] = 1.0;
    PressureUnitConvertorDic[PressureUnit::Bar][PressureUnit::Dyne_Centimeter2] = Bar__T__Pascal * 1.0 / Dyne_Centimeter2__T__Pascal;

    PressureUnitConvertorDic[PressureUnit::Dyne_Centimeter2][PressureUnit::Pascal] = Dyne_Centimeter2__T__Pascal;
    PressureUnitConvertorDic[PressureUnit::Dyne_Centimeter2][PressureUnit::Atmosphere] = Dyne_Centimeter2__T__Pascal * 1.0 / Atmosphere__T__Pascal;
    PressureUnitConvertorDic[PressureUnit::Dyne_Centimeter2][PressureUnit::Bar] = Dyne_Centimeter2__T__Pascal * 1.0 / Bar__T__Pascal;
    PressureUnitConvertorDic[PressureUnit::Dyne_Centimeter2][PressureUnit::Dyne_Centimeter2] = 1.0;

    ViscosityUnitConvertorDic[ViscosityUnit::PascalSecond][ViscosityUnit::PascalSecond] = 1.0;
    ViscosityUnitConvertorDic[ViscosityUnit::PascalSecond][ViscosityUnit::Poise] = 1.0 / Poise__T__PascalSecond;
    ViscosityUnitConvertorDic[ViscosityUnit::PascalSecond][ViscosityUnit::Amu_BohrFemtosecond] = 1.0 / Amu_BohrFemtosecond__T__PascalSecond;

    ViscosityUnitConvertorDic[ViscosityUnit::Poise][ViscosityUnit::PascalSecond] = Poise__T__PascalSecond;
    ViscosityUnitConvertorDic[ViscosityUnit::Poise][ViscosityUnit::Poise] = 1.0;
    ViscosityUnitConvertorDic[ViscosityUnit::Poise][ViscosityUnit::Amu_BohrFemtosecond] = Poise__T__PascalSecond * 1.0 / Amu_BohrFemtosecond__T__PascalSecond;

    ViscosityUnitConvertorDic[ViscosityUnit::Amu_BohrFemtosecond][ViscosityUnit::PascalSecond] = Amu_BohrFemtosecond__T__PascalSecond;
    ViscosityUnitConvertorDic[ViscosityUnit::Amu_BohrFemtosecond][ViscosityUnit::Poise] = Amu_BohrFemtosecond__T__PascalSecond * 1.0 / Poise__T__PascalSecond;
    ViscosityUnitConvertorDic[ViscosityUnit::Amu_BohrFemtosecond][ViscosityUnit::Amu_BohrFemtosecond] = 1.0;

    ChargeUnitConvertorDic[ChargeUnit::Coulomb][ChargeUnit::Coulomb] = 1.0;
    ChargeUnitConvertorDic[ChargeUnit::Coulomb][ChargeUnit::Electron] = 1.0 / Electron__T__Coulomb;
    ChargeUnitConvertorDic[ChargeUnit::Coulomb][ChargeUnit::StatCoulomb] = 1.0 / StatCoulomb__T__Coulomb;

    ChargeUnitConvertorDic[ChargeUnit::Electron][ChargeUnit::Coulomb] = Electron__T__Coulomb;
    ChargeUnitConvertorDic[ChargeUnit::Electron][ChargeUnit::Electron] = 1.0;
    ChargeUnitConvertorDic[ChargeUnit::Electron][ChargeUnit::StatCoulomb] = Electron__T__Coulomb * 1.0 / StatCoulomb__T__Coulomb;

    ChargeUnitConvertorDic[ChargeUnit::StatCoulomb][ChargeUnit::Coulomb] = StatCoulomb__T__Coulomb;
    ChargeUnitConvertorDic[ChargeUnit::StatCoulomb][ChargeUnit::Electron] = StatCoulomb__T__Coulomb * 1.0 / Electron__T__Coulomb;
    ChargeUnitConvertorDic[ChargeUnit::StatCoulomb][ChargeUnit::StatCoulomb] = 1.0;

    DipoleUnitConvertorDic[DipoleUnit::CoulombMeter][DipoleUnit::CoulombMeter] = 1.0;
    DipoleUnitConvertorDic[DipoleUnit::CoulombMeter][DipoleUnit::ElectronAngstrom] = 1.0 / ElectronAngstrom__T__CoulombMeter;
    DipoleUnitConvertorDic[DipoleUnit::CoulombMeter][DipoleUnit::StatCoulombCentimeter] = 1.0 / StatCoulombCentimeter__T__CoulombMeter;
    DipoleUnitConvertorDic[DipoleUnit::CoulombMeter][DipoleUnit::Debye] = 1.0 / Debye__T__CoulombMeter;

    DipoleUnitConvertorDic[DipoleUnit::ElectronAngstrom][DipoleUnit::CoulombMeter] = ElectronAngstrom__T__CoulombMeter;
    DipoleUnitConvertorDic[DipoleUnit::ElectronAngstrom][DipoleUnit::ElectronAngstrom] = 1.0;
    DipoleUnitConvertorDic[DipoleUnit::ElectronAngstrom][DipoleUnit::StatCoulombCentimeter] = ElectronAngstrom__T__CoulombMeter * 1.0 / StatCoulombCentimeter__T__CoulombMeter;
    DipoleUnitConvertorDic[DipoleUnit::ElectronAngstrom][DipoleUnit::Debye] = ElectronAngstrom__T__CoulombMeter * 1.0 / Debye__T__CoulombMeter;

    DipoleUnitConvertorDic[DipoleUnit::StatCoulombCentimeter][DipoleUnit::CoulombMeter] = StatCoulombCentimeter__T__CoulombMeter;
    DipoleUnitConvertorDic[DipoleUnit::StatCoulombCentimeter][DipoleUnit::ElectronAngstrom] = StatCoulombCentimeter__T__CoulombMeter * 1.0 / ElectronAngstrom__T__CoulombMeter;
    DipoleUnitConvertorDic[DipoleUnit::StatCoulombCentimeter][DipoleUnit::StatCoulombCentimeter] = 1.0;
    DipoleUnitConvertorDic[DipoleUnit::StatCoulombCentimeter][DipoleUnit::Debye] = StatCoulombCentimeter__T__CoulombMeter * 1.0 / Debye__T__CoulombMeter;

    DipoleUnitConvertorDic[DipoleUnit::Debye][DipoleUnit::CoulombMeter] = Debye__T__CoulombMeter;
    DipoleUnitConvertorDic[DipoleUnit::Debye][DipoleUnit::ElectronAngstrom] = Debye__T__CoulombMeter * 1.0 / ElectronAngstrom__T__CoulombMeter;
    DipoleUnitConvertorDic[DipoleUnit::Debye][DipoleUnit::StatCoulombCentimeter] = Debye__T__CoulombMeter * 1.0 / StatCoulombCentimeter__T__CoulombMeter;
    DipoleUnitConvertorDic[DipoleUnit::Debye][DipoleUnit::Debye] = 1.0;

    ElectricFieldUnitConvertorDic[ElectricFieldUnit::Volt_Meter][ElectricFieldUnit::Volt_Meter] = 1.0;
    ElectricFieldUnitConvertorDic[ElectricFieldUnit::Volt_Meter][ElectricFieldUnit::Volt_Angstrom] = 1.0 / Volt_Angstrom__T__Volt_Meter;
    ElectricFieldUnitConvertorDic[ElectricFieldUnit::Volt_Meter][ElectricFieldUnit::StatVolt_Centimeter] = 1.0 / StatVolt_Centimeter__T__Volt_Meter;
    ElectricFieldUnitConvertorDic[ElectricFieldUnit::Volt_Meter][ElectricFieldUnit::Volt_Centimeter] = 1.0 / Volt_Centimeter__T__Volt_Meter;

    ElectricFieldUnitConvertorDic[ElectricFieldUnit::Volt_Angstrom][ElectricFieldUnit::Volt_Meter] = Volt_Angstrom__T__Volt_Meter;
    ElectricFieldUnitConvertorDic[ElectricFieldUnit::Volt_Angstrom][ElectricFieldUnit::Volt_Angstrom] = 1.0;
    ElectricFieldUnitConvertorDic[ElectricFieldUnit::Volt_Angstrom][ElectricFieldUnit::StatVolt_Centimeter] = Volt_Angstrom__T__Volt_Meter * 1.0 / StatVolt_Centimeter__T__Volt_Meter;
    ElectricFieldUnitConvertorDic[ElectricFieldUnit::Volt_Angstrom][ElectricFieldUnit::Volt_Centimeter] = Volt_Angstrom__T__Volt_Meter * 1.0 / Volt_Centimeter__T__Volt_Meter;

    ElectricFieldUnitConvertorDic[ElectricFieldUnit::StatVolt_Centimeter][ElectricFieldUnit::Volt_Meter] = StatVolt_Centimeter__T__Volt_Meter;
    ElectricFieldUnitConvertorDic[ElectricFieldUnit::StatVolt_Centimeter][ElectricFieldUnit::Volt_Angstrom] = StatVolt_Centimeter__T__Volt_Meter * 1.0 / Volt_Angstrom__T__Volt_Meter;
    ElectricFieldUnitConvertorDic[ElectricFieldUnit::StatVolt_Centimeter][ElectricFieldUnit::StatVolt_Centimeter] = 1.0;
    ElectricFieldUnitConvertorDic[ElectricFieldUnit::StatVolt_Centimeter][ElectricFieldUnit::Volt_Centimeter] = StatVolt_Centimeter__T__Volt_Meter * 1.0 / Volt_Centimeter__T__Volt_Meter;

    ElectricFieldUnitConvertorDic[ElectricFieldUnit::Volt_Centimeter][ElectricFieldUnit::Volt_Meter] = Volt_Centimeter__T__Volt_Meter;
    ElectricFieldUnitConvertorDic[ElectricFieldUnit::Volt_Centimeter][ElectricFieldUnit::Volt_Angstrom] = Volt_Centimeter__T__Volt_Meter * 1.0 / Volt_Angstrom__T__Volt_Meter;
    ElectricFieldUnitConvertorDic[ElectricFieldUnit::Volt_Centimeter][ElectricFieldUnit::StatVolt_Centimeter] = Volt_Centimeter__T__Volt_Meter * 1.0 / StatVolt_Centimeter__T__Volt_Meter;
    ElectricFieldUnitConvertorDic[ElectricFieldUnit::Volt_Centimeter][ElectricFieldUnit::Volt_Centimeter] = 1.0;

    DensityUnitConvertorDic[DensityUnit::kGram_Meter3][DensityUnit::kGram_Meter3] = 1.0;
    DensityUnitConvertorDic[DensityUnit::kGram_Meter3][DensityUnit::Gram_Centimeter3] = 1.0 / Gram_Centimeter3__T__kGram_Meter3;
    DensityUnitConvertorDic[DensityUnit::kGram_Meter3][DensityUnit::Amu_Bohr3] = 1.0 / Amu_Bohr3__T__kGram_Meter3;

    DensityUnitConvertorDic[DensityUnit::Gram_Centimeter3][DensityUnit::kGram_Meter3] = Gram_Centimeter3__T__kGram_Meter3;
    DensityUnitConvertorDic[DensityUnit::Gram_Centimeter3][DensityUnit::Gram_Centimeter3] = 1.0;
    DensityUnitConvertorDic[DensityUnit::Gram_Centimeter3][DensityUnit::Amu_Bohr3] = Gram_Centimeter3__T__kGram_Meter3 * 1.0 / Amu_Bohr3__T__kGram_Meter3;

    DensityUnitConvertorDic[DensityUnit::Amu_Bohr3][DensityUnit::kGram_Meter3] = Amu_Bohr3__T__kGram_Meter3;
    DensityUnitConvertorDic[DensityUnit::Amu_Bohr3][DensityUnit::Gram_Centimeter3] = Amu_Bohr3__T__kGram_Meter3 * 1.0 / Gram_Centimeter3__T__kGram_Meter3;
    DensityUnitConvertorDic[DensityUnit::Amu_Bohr3][DensityUnit::Amu_Bohr3] = 1.0;

    dictionaries_are_initialized = true;
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

    init_dictionaries();

    massUnitScale = MassUnitConvertorDic[MassUnitDic[fromStyle]][MassUnitDic[unitStyle]];
    lengthUnitScale = LengthUnitConvertorDic[LengthUnitDic[fromStyle]][LengthUnitDic[unitStyle]];
    timeUnitScale = TimeUnitConvertorDic[TimeUnitDic[fromStyle]][TimeUnitDic[unitStyle]];
    energyUnitScale = EnergyUnitConvertorDic[EnergyUnitDic[fromStyle]][EnergyUnitDic[unitStyle]];
    velocityUnitScale = VelocityUnitConvertorDic[VelocityUnitDic[fromStyle]][VelocityUnitDic[unitStyle]];
    forceUnitScale = ForceUnitConvertorDic[ForceUnitDic[fromStyle]][ForceUnitDic[unitStyle]];
    torqueUnitScale = TorqueUnitConvertorDic[TorqueUnitDic[fromStyle]][TorqueUnitDic[unitStyle]];
    temperatureUnitScale = TemperatureUnitConvertorDic[TemperatureUnitDic[fromStyle]][TemperatureUnitDic[unitStyle]];
    pressureUnitScale = PressureUnitConvertorDic[PressureUnitDic[fromStyle]][PressureUnitDic[unitStyle]];
    viscosityUnitScale = ViscosityUnitConvertorDic[ViscosityUnitDic[fromStyle]][ViscosityUnitDic[unitStyle]];
    chargeUnitScale = ChargeUnitConvertorDic[ChargeUnitDic[fromStyle]][ChargeUnitDic[unitStyle]];
    dipoleUnitScale = DipoleUnitConvertorDic[DipoleUnitDic[fromStyle]][DipoleUnitDic[unitStyle]];
    electricFieldUnitScale = ElectricFieldUnitConvertorDic[ElectricFieldUnitDic[fromStyle]][ElectricFieldUnitDic[unitStyle]];
    densityUnitScale = DensityUnitConvertorDic[DensityUnitDic[fromStyle]][DensityUnitDic[unitStyle]];

    return true;
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

    init_dictionaries();

    massUnitScale = MassUnitConvertorDic[MassUnitDic[unitStyle]][MassUnitDic[toStyle]];
    lengthUnitScale = LengthUnitConvertorDic[LengthUnitDic[unitStyle]][LengthUnitDic[toStyle]];
    timeUnitScale = TimeUnitConvertorDic[TimeUnitDic[unitStyle]][TimeUnitDic[toStyle]];
    energyUnitScale = EnergyUnitConvertorDic[EnergyUnitDic[unitStyle]][EnergyUnitDic[toStyle]];
    velocityUnitScale = VelocityUnitConvertorDic[VelocityUnitDic[unitStyle]][VelocityUnitDic[toStyle]];
    forceUnitScale = ForceUnitConvertorDic[ForceUnitDic[unitStyle]][ForceUnitDic[toStyle]];
    torqueUnitScale = TorqueUnitConvertorDic[TorqueUnitDic[unitStyle]][TorqueUnitDic[toStyle]];
    temperatureUnitScale = TemperatureUnitConvertorDic[TemperatureUnitDic[unitStyle]][TemperatureUnitDic[toStyle]];
    pressureUnitScale = PressureUnitConvertorDic[PressureUnitDic[unitStyle]][PressureUnitDic[toStyle]];
    viscosityUnitScale = ViscosityUnitConvertorDic[ViscosityUnitDic[unitStyle]][ViscosityUnitDic[toStyle]];
    chargeUnitScale = ChargeUnitConvertorDic[ChargeUnitDic[unitStyle]][ChargeUnitDic[toStyle]];
    dipoleUnitScale = DipoleUnitConvertorDic[DipoleUnitDic[unitStyle]][DipoleUnitDic[toStyle]];
    electricFieldUnitScale = ElectricFieldUnitConvertorDic[ElectricFieldUnitDic[unitStyle]][ElectricFieldUnitDic[toStyle]];
    densityUnitScale = DensityUnitConvertorDic[DensityUnitDic[unitStyle]][DensityUnitDic[toStyle]];

    return true;
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

template <UnitType UNIT>
inline void units::convert(double &value)
{
    UMUQFAIL("This is not implemented on purpose!");
}

template <>
inline void units::convert<UnitType::Mass>(double &value)
{
    value *= massUnitScale;
}

template <>
inline void units::convert<UnitType::Length>(double &value)
{
    value *= lengthUnitScale;
}

template <>
inline void units::convert<UnitType::Time>(double &value)
{
    value *= timeUnitScale;
}

template <>
inline void units::convert<UnitType::Energy>(double &value)
{
    value *= energyUnitScale;
}

template <>
inline void units::convert<UnitType::Velocity>(double &value)
{
    value *= velocityUnitScale;
}

template <>
inline void units::convert<UnitType::Force>(double &value)
{
    value *= forceUnitScale;
}

template <>
inline void units::convert<UnitType::Torque>(double &value)
{
    value *= torqueUnitScale;
}

template <>
inline void units::convert<UnitType::Temperature>(double &value)
{
    value *= temperatureUnitScale;
}

template <>
inline void units::convert<UnitType::Pressure>(double &value)
{
    value *= pressureUnitScale;
}

template <>
inline void units::convert<UnitType::Viscosity>(double &value)
{
    value *= viscosityUnitScale;
}

template <>
inline void units::convert<UnitType::Charge>(double &value)
{
    value *= chargeUnitScale;
}

template <>
inline void units::convert<UnitType::Dipole>(double &value)
{
    value *= dipoleUnitScale;
}

template <>
inline void units::convert<UnitType::ElectricField>(double &value)
{
    value *= electricFieldUnitScale;
}

template <>
inline void units::convert<UnitType::Density>(double &value)
{
    value *= densityUnitScale;
}

template <UnitType UNIT>
inline void units::convert(std::vector<double> &value)
{
    UMUQFAIL("This is not implemented on purpose!");
}

template <>
inline void units::convert<UnitType::Mass>(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= massUnitScale; });
}

template <>
inline void units::convert<UnitType::Length>(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= lengthUnitScale; });
}

template <>
inline void units::convert<UnitType::Time>(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= timeUnitScale; });
}

template <>
inline void units::convert<UnitType::Energy>(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= energyUnitScale; });
}

template <>
inline void units::convert<UnitType::Velocity>(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= velocityUnitScale; });
}

template <>
inline void units::convert<UnitType::Force>(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= forceUnitScale; });
}

template <>
inline void units::convert<UnitType::Torque>(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= torqueUnitScale; });
}

template <>
inline void units::convert<UnitType::Temperature>(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= temperatureUnitScale; });
}

template <>
inline void units::convert<UnitType::Pressure>(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= pressureUnitScale; });
}

template <>
inline void units::convert<UnitType::Viscosity>(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= viscosityUnitScale; });
}

template <>
inline void units::convert<UnitType::Charge>(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= chargeUnitScale; });
}

template <>
inline void units::convert<UnitType::Dipole>(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= dipoleUnitScale; });
}

template <>
inline void units::convert<UnitType::ElectricField>(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= electricFieldUnitScale; });
}

template <>
inline void units::convert<UnitType::Density>(std::vector<double> &value)
{
    std::for_each(value.begin(), value.end(), [&](double &v) { v *= densityUnitScale; });
}

template <UnitType UNIT>
inline void units::convert(double *value, int const nSize)
{
    UMUQFAIL("This is not implemented on purpose!");
}

template <>
inline void units::convert<UnitType::Mass>(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= massUnitScale; });
}

template <>
inline void units::convert<UnitType::Length>(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= lengthUnitScale; });
}

template <>
inline void units::convert<UnitType::Time>(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= timeUnitScale; });
}

template <>
inline void units::convert<UnitType::Energy>(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= energyUnitScale; });
}

template <>
inline void units::convert<UnitType::Velocity>(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= velocityUnitScale; });
}

template <>
inline void units::convert<UnitType::Force>(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= forceUnitScale; });
}

template <>
inline void units::convert<UnitType::Torque>(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= torqueUnitScale; });
}

template <>
inline void units::convert<UnitType::Temperature>(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= temperatureUnitScale; });
}

template <>
inline void units::convert<UnitType::Pressure>(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= pressureUnitScale; });
}

template <>
inline void units::convert<UnitType::Viscosity>(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= viscosityUnitScale; });
}

template <>
inline void units::convert<UnitType::Charge>(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= chargeUnitScale; });
}

template <>
inline void units::convert<UnitType::Dipole>(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= dipoleUnitScale; });
}

template <>
inline void units::convert<UnitType::ElectricField>(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= electricFieldUnitScale; });
}

template <>
inline void units::convert<UnitType::Density>(double *value, int const nSize)
{
    std::for_each(value, value + nSize, [&](double &v) { v *= densityUnitScale; });
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
    if (fromStyle == toStyle)
    {
        UMUQWARNING("The old style == the current style == ", umuq::getUnitStyleName(fromStyle), " and nothing will change!");
        return true;
    }
    umuq::units u(toStyle);
    if (u.convertFromStyle(fromStyle))
    {
        if (std::is_same<UNIT, umuq::MassUnit>::value)
        {
            u.convert<UnitType::Mass>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::LengthUnit>::value)
        {
            u.convert<UnitType::Length>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::TimeUnit>::value)
        {
            u.convert<UnitType::Time>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::EnergyUnit>::value)
        {
            u.convert<UnitType::Energy>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::VelocityUnit>::value)
        {
            u.convert<UnitType::Velocity>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::ForceUnit>::value)
        {
            u.convert<UnitType::Force>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::TorqueUnit>::value)
        {
            u.convert<UnitType::Torque>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::TemperatureUnit>::value)
        {
            u.convert<UnitType::Temperature>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::PressureUnit>::value)
        {
            u.convert<UnitType::Pressure>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::ViscosityUnit>::value)
        {
            u.convert<UnitType::Viscosity>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::ChargeUnit>::value)
        {
            u.convert<UnitType::Charge>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::DipoleUnit>::value)
        {
            u.convert<UnitType::Dipole>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::ElectricFieldUnit>::value)
        {
            u.convert<UnitType::ElectricField>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::DensityUnit>::value)
        {
            u.convert<UnitType::Density>(value);
            return true;
        }
        UMUQWARNING("This is not a valid UNIT");
        return false;
    }
    return false;
}

template <umuq::UnitType UNIT>
bool convert(std::vector<double> &value, UnitStyle const &fromStyle, UnitStyle const &toStyle)
{
    if (fromStyle == toStyle)
    {
        UMUQWARNING("The old style == the current style == ", umuq::getUnitStyleName(fromStyle), " and nothing will change!");
        return true;
    }
    umuq::units u(toStyle);
    if (u.convertFromStyle(fromStyle))
    {
        switch (UNIT)
        {
        case (umuq::UnitType::Mass):
            u.convert<UnitType::Mass>(value);
            break;
        case (umuq::UnitType::Length):
            u.convert<UnitType::Length>(value);
            break;
        case (umuq::UnitType::Time):
            u.convert<UnitType::Time>(value);
            break;
        case (umuq::UnitType::Energy):
            u.convert<UnitType::Energy>(value);
            break;
        case (umuq::UnitType::Velocity):
            u.convert<UnitType::Velocity>(value);
            break;
        case (umuq::UnitType::Force):
            u.convert<UnitType::Force>(value);
            break;
        case (umuq::UnitType::Torque):
            u.convert<UnitType::Torque>(value);
            break;
        case (umuq::UnitType::Temperature):
            u.convert<UnitType::Temperature>(value);
            break;
        case (umuq::UnitType::Pressure):
            u.convert<UnitType::Pressure>(value);
            break;
        case (umuq::UnitType::Viscosity):
            u.convert<UnitType::Viscosity>(value);
            break;
        case (umuq::UnitType::Charge):
            u.convert<UnitType::Charge>(value);
            break;
        case (umuq::UnitType::Dipole):
            u.convert<UnitType::Dipole>(value);
            break;
        case (umuq::UnitType::ElectricField):
            u.convert<UnitType::ElectricField>(value);
            break;
        case (umuq::UnitType::Density):
            u.convert<UnitType::Density>(value);
            break;
        default:
            UMUQWARNING("This is not a valid UNIT");
            return false;
            break;
        }
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
    if (fromStyle == toStyle)
    {
        UMUQWARNING("The old style == the current style == ", fromStyle, " and nothing will change!");
        return true;
    }
    umuq::units u(toStyle);
    if (u.convertFromStyle(fromStyle))
    {
        if (std::is_same<UNIT, umuq::MassUnit>::value)
        {
            u.convert<UnitType::Mass>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::LengthUnit>::value)
        {
            u.convert<UnitType::Length>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::TimeUnit>::value)
        {
            u.convert<UnitType::Time>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::EnergyUnit>::value)
        {
            u.convert<UnitType::Energy>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::VelocityUnit>::value)
        {
            u.convert<UnitType::Velocity>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::ForceUnit>::value)
        {
            u.convert<UnitType::Force>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::TorqueUnit>::value)
        {
            u.convert<UnitType::Torque>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::TemperatureUnit>::value)
        {
            u.convert<UnitType::Temperature>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::PressureUnit>::value)
        {
            u.convert<UnitType::Pressure>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::ViscosityUnit>::value)
        {
            u.convert<UnitType::Viscosity>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::ChargeUnit>::value)
        {
            u.convert<UnitType::Charge>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::DipoleUnit>::value)
        {
            u.convert<UnitType::Dipole>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::ElectricFieldUnit>::value)
        {
            u.convert<UnitType::ElectricField>(value);
            return true;
        }
        else if (std::is_same<UNIT, umuq::DensityUnit>::value)
        {
            u.convert<UnitType::Density>(value);
            return true;
        }
        UMUQWARNING("This is not a valid UNIT");
        return false;
    }
    return false;
}

template <umuq::UnitType UNIT>
bool convert(std::vector<double> &value, std::string const &fromStyle, std::string const &toStyle)
{
    if (fromStyle == toStyle)
    {
        UMUQWARNING("The old style == the current style == ", fromStyle, " and nothing will change!");
        return true;
    }
    umuq::units u(toStyle);
    if (u.convertFromStyle(fromStyle))
    {
        switch (UNIT)
        {
        case (umuq::UnitType::Mass):
            u.convert<UnitType::Mass>(value);
            break;
        case (umuq::UnitType::Length):
            u.convert<UnitType::Length>(value);
            break;
        case (umuq::UnitType::Time):
            u.convert<UnitType::Time>(value);
            break;
        case (umuq::UnitType::Energy):
            u.convert<UnitType::Energy>(value);
            break;
        case (umuq::UnitType::Velocity):
            u.convert<UnitType::Velocity>(value);
            break;
        case (umuq::UnitType::Force):
            u.convert<UnitType::Force>(value);
            break;
        case (umuq::UnitType::Torque):
            u.convert<UnitType::Torque>(value);
            break;
        case (umuq::UnitType::Temperature):
            u.convert<UnitType::Temperature>(value);
            break;
        case (umuq::UnitType::Pressure):
            u.convert<UnitType::Pressure>(value);
            break;
        case (umuq::UnitType::Viscosity):
            u.convert<UnitType::Viscosity>(value);
            break;
        case (umuq::UnitType::Charge):
            u.convert<UnitType::Charge>(value);
            break;
        case (umuq::UnitType::Dipole):
            u.convert<UnitType::Dipole>(value);
            break;
        case (umuq::UnitType::ElectricField):
            u.convert<UnitType::ElectricField>(value);
            break;
        case (umuq::UnitType::Density):
            u.convert<UnitType::Density>(value);
            break;
        default:
            UMUQWARNING("This is not a valid UNIT");
            return false;
        }
        return true;
    }
    return false;
}

} // namespace umuq

#endif // UMUQ_UNITS
