/**
 * \file se2_sam.cpp
 *
 *  Created on: Feb 4, 2019
 *      \author: jsola
 *
 *  ------------------------------------------------------------
 *  This file is:
 *  (c) 2018 Joan Sola @ IRI-CSIC, Barcelona, Catalonia
 *
 *  This file is part of `manif`, a C++ template-only library
 *  for Lie theory targeted at estimation for robotics.
 *  Manif is:
 *  (c) 2018 Jeremie Deray @ IRI-UPC, Barcelona
 *  ------------------------------------------------------------
 *
 *  ------------------------------------------------------------
 *  Demonstration example:
 *
 *  2D smoothing and mapping (SAM).
 *
 *  See se3_sam.cpp          for a 3D version of this example.
 *  See se2_localization.cpp for a simpler localization example using EKF.
 *  ------------------------------------------------------------
 *
 *  This demo corresponds to the application
 *  in chapter V, section B, in the paper Sola-18,
 *  [https://arxiv.org/abs/1812.01537].
 *
 *  The following is an abstract of the content of the paper.
 *  Please consult the paper for better reference.
 *
 *
 *  We consider a robot in 2D space surrounded by a small
 *  number of punctual landmarks or _beacons_.
 *  The robot receives control actions in the form of axial
 *  and angular velocities, and is able to measure the location
 *  of the beacons w.r.t its own reference frame.
 *
 *  The robot pose X_i is in SE(2) and the beacon positions b_k in R^2,
 *
 *      X_i = |  R_i   t_i |        // position and orientation
 *            |   0     1  |
 *
 *      b_k = (bx_k, by_k)          // lmk coordinates in world frame
 *
 *  The control signal u is a twist in se(2) comprising longitudinal
 *  velocity vx and angular velocity wz, with no other velocity
 *  components, integrated over the sampling time dt.
 *
 *      u = (vx*dt, 0, w*dt)
 *
 *  The control is corrupted by additive Gaussian noise u_noise,
 *  with covariance
 *
 *      Q = diagonal(sigma_v^2, sigma_s^2, sigma_yaw^2).
 *
 *  This noise accounts for possible lateral slippage
 *  through non-zero values of sigma_s.
 *
 *  At the arrival of a control u, a new robot pose is created at
 *
 *      X_j = X_i * Exp(u) = X_i + u.
 *
 *  This new pose is then added to the graph.
 *
 *  Landmark measurements are of the range and bearing type,
 *  though they are put in Cartesian form for simplicity,
 *
 *      y = (yx, yy)           // lmk coordinates in robot frame
 *
 *  Their noise n is zero mean Gaussian, and is specified
 *  with a covariances matrix R.
 *  We notice the rigid motion action y_ik = h(X_i,b_k) = X_i^-1 * b_k
 *  (see appendix D).
 *
 *
 *  The world comprises 5 landmarks.
 *  Not all of them are observed from each pose.
 *  A set of pairs pose--landmark is created to establish which
 *  landmarks are observed from each pose.
 *  These pairs can be observed in the factor graph, as follows.
 *
 *  The factor graph of the SAM problem looks like this:
 *
 *                  ------- b1
 *          b3    /         |
 *          |   /       b4  |
 *          | /       /    \|
 *          X0 ---- X1 ---- X2
 *          | \   /   \   /
 *          |   b0      b2
 *          *
 *
 *  where:
 *    - X_i are SE2 robot poses_rob1
 *    - b_k are R^2 landmarks or beacons
 *    - * is a pose prior to anchor the map and make the problem observable
 *    - segments indicate measurement factors:
 *      - motion measurements from X_i to X_j
 *      - landmark measurements from X_i to b_k
 *      - absolute pose measurement from X0 to * (the origin of coordinates)
 *
 *  We thus declare 9 factors pose---landmark, as follows:
 *
 *    poses_rob1 ---  lmks
 *      x0  ---  b0
 *      x0  ---  b1
 *      x0  ---  b3
 *      x1  ---  b0
 *      x1  ---  b2
 *      x1  ---  b4
 *      x2  ---  b1
 *      x2  ---  b2
 *      x2  ---  b4
 *
 *
 *  The main variables are summarized again as follows
 *
 *      Xi_rob1  : robot pose at time i, SE(2)
 *      u   : robot control, (v*dt; 0; w*dt) in se(2)
 *      Q   : control perturbation covariance
 *      b   : landmark position, R^2
 *      y   : Cartesian landmark measurement in robot frame, R^2
 *      R   : covariance of the measurement noise
 *
 *
 *  We define the state to estimate as a manifold composite:
 *
 *      X in  < SE2, SE2, SE2, R^2, R^2, R^2, R^2, R^2 >
 *
 *      X  =  <  X0,  X1,  X2,  b0,  b1,  b2,  b3,  b4 >
 *
 *  The estimation error dX is expressed
 *  in the tangent space at X,
 *
 *      dX in  < se2, se2, se2, R^2, R^2, R^2, R^2, R^2 >
 *          ~  < R^3, R^3, R^3, R^2, R^2, R^2, R^2, R^2 > = R^19
 *
 *      dX  =  [ dx0, dx1, dx2, db0, db1, db2, db3, db4 ] in R^19
 *
 *  with
 *      dx_i: pose error in se(2) ~ R^3
 *      db_k: landmark error in R^2
 *
 *
 *  The prior, motion and measurement models are
 *
 *    - for the prior factor:
 *        p_0     = X_0
 *
 *    - for the motion factors:
 *        d_ij    = X_j (-) X_i = log(X_i.inv * X_j)  // motion expectation equation
 *
 *    - for the measurement factors:
 *        e_ik    = h(X_i, b_k) = X_i^-1 * b_k        // measurement expectation equation
 *
 *
 *
 *  The algorithm below comprises first a simulator to
 *  produce measurements, then uses these measurements
 *  to estimate the state, using a graph representation
 *  and Lie-based non-linear iterative least squares solver
 *  that uses the pseudo-inverse method.
 *
 *  This file has plain code with only one main() function.
 *  There are no function calls other than those involving `manif`.
 *
 *  Printing the prior state (before solving) and posterior state (after solving),
 *  together with a ground-truth state defined by the simulator
 *  allows for evaluating the quality of the estimates.
 *
 *  This information is complemented with the evolution of
 *  the optimizer's residual and optimal step norms. This allows
 *  for evaluating the convergence of the optimizer.
 */



// manif
#include "manif/SE2.h"

// Std
#include <vector>
#include <map>
#include <list>
#include <cstdlib>

// Debug
#include <iostream>
#include <iomanip>

// std shortcuts and namespaces
using std::cout;
using std::endl;
using std::vector;
using std::map;
using std::list;
using std::pair;

// Eigen namespace
using namespace Eigen;

// manif namespace and shortcuts
using manif::SE2d;
using manif::SE2Tangentd;

// Navid: Dim=2 and Dof=3
static constexpr int DoF = SE2d::DoF;
static constexpr int Dim = SE2d::Dim;

// Define many data types (Tangent refers to the tangent of SE2)
typedef Array<double,  DoF, 1>      ArrayT;     // tangent-size array
typedef Matrix<double, DoF, 1>      VectorT;    // tangent-size vector
typedef Matrix<double, DoF, DoF>    MatrixT;    // tangent-size square matrix
typedef Matrix<double, Dim, 1>      VectorB;    // landmark-size vector
typedef Array<double,  Dim, 1>      ArrayY;     // measurement-size array
typedef Matrix<double, Dim, 1>      VectorY;    // measurement-size vector
typedef Matrix<double, Dim, Dim>    MatrixY;    // measurement-size square matrix
typedef Matrix<double, Dim, DoF>    MatrixYT;   // measurement x tangent size matrix
typedef Matrix<double, Dim, Dim>    MatrixYB;   // measurement x landmark size matrix

// some experiment constants
static const int NUM_POSES      = 3;
static const int NUM_LMKS       = 5;
static const int NUM_FACTORS    = 9;
static const int NUM_STATES     = NUM_POSES * DoF + NUM_LMKS    * Dim;
static const int NUM_MEAS       = NUM_POSES * DoF + NUM_FACTORS * Dim;
static const int MAX_ITER       = 20;           // for the solver

int main()
{
    std::srand((unsigned int) time(0));

    // DEBUG INFO
    cout << endl;
    cout << "2D Smoothing and Mapping. 3 poses_rob1, 5 landmarks." << endl;
    cout << "-----------------------------------------------" << endl;
    cout << std::fixed   << std::setprecision(3) << std::showpos;

    // START CONFIGURATION
    //
    //

    // Define the robot pose elements
    SE2d         X_simu_rob1,    // pose of the simulated robot 1
                 Xi_rob1,        // robot pose at time i
                 Xj_rob1,
                 X_simu_rob2(2.0, 2.0, 1, 0),
                 Xi_rob2(2.0, 2.0, 1 ,0),
                 Xj_rob2,
                 tf_(2.0, 2.0, 1 ,0),
                 tf_haniye;        // robot pose at time j
    vector<SE2d> poses_rob1,     // estimator poses_rob1
                 poses_simu_rob1,
                 poses_rob2,
                 poses_simu_rob2;// simulator poses_rob1
    
    
    Xi_rob1.setIdentity();
    X_simu_rob1.setIdentity();

    float sc=0.1;

    // Define a control vector and its noise and covariance in the tangent of SE2
    SE2Tangentd         u1, u2;          // control signal, generic
    SE2Tangentd         u_nom_rob1, u_nom_rob2;      // nominal control signal
    ArrayT              u_sigmas, los_sigmas;   // control noise std specification
    VectorT             u_noise;    // control noise
    MatrixT             Q;          // Covariance
    MatrixT             W, W_los;          // sqrt Info
    vector<SE2Tangentd> controls1, controls2;   // robot controls

    u_nom_rob1     << 0.1, 0.0, 0.5;
    u_nom_rob2     << 0.2, 0.0, 1;

    u_sigmas  << 0.8*sc, 0.8*sc, 0.8*sc;
    los_sigmas << 0.001, 0.001, 0.001;
    // Q         = (u_sigmas * u_sigmas).matrix().asDiagonal();
    W         =  u_sigmas.inverse()  .matrix().asDiagonal(); // this is Q^(-T/2)
    W_los         =  los_sigmas.inverse()  .matrix().asDiagonal(); // this is Q^(-T/2)

    // Landmarks in R^2 and map
    VectorB b1, b2; // Landmark, generic
    vector<VectorB> landmarks1(NUM_LMKS), landmarks_simu;
    {
        // Define five landmarks (beacons) in R^2
        VectorB b0, b1, b2, b3, b4;
        b0 << 3.0,  0.0;
        b1 << 2.0, -1.0;
        b2 << 2.0,  1.0;
        b3 << 3.0, -1.0;
        b4 << 3.0,  1.0;
        landmarks_simu.push_back(b0);
        landmarks_simu.push_back(b1);
        landmarks_simu.push_back(b2);
        landmarks_simu.push_back(b3);
        landmarks_simu.push_back(b4);
    } // destroy b0...b4




    vector<VectorB> landmarks2(NUM_LMKS);
    {
        // Define five landmarks (beacons) in R^2
        VectorB b0, b1, b2, b3, b4;
        b0 << 3.0,  0.0;
        b1 << 2.0, -1.0;
        b2 << 2.0,  1.0;
        b3 << 3.0, -1.0;
        b4 << 3.0,  1.0;

    } // destroy b0...b4

    // Define the beacon's measurements in R^2
    VectorY             y1, y2, y_noise;
    ArrayY              y_sigmas;
    VectorT             los_noise;
    VectorT             temp;
    MatrixY             R; // Covariance
    MatrixY             S; // sqrt Info
    vector<map<int,VectorY>> measurements1(NUM_POSES); // y = measurements[pose_id][lmk_id]
    vector<map<int,VectorY>> measurements2(NUM_POSES); // y = measurements[pose_id][lmk_id]

    y_sigmas << 0.08*sc, 0.08*sc;
    los_sigmas << 0.001, 0.001, 0.001;
    // R        = (y_sigmas * y_sigmas).matrix().asDiagonal();
    S        =  y_sigmas.inverse()  .matrix().asDiagonal(); // this is R^(-T/2)

    // Declare some temporaries
    SE2Tangentd     d1, d2, d_parham, d_arash;              // motion expectation d = Xj_rob1 (-) Xi_rob1 = Xj_rob1.minus ( Xi_rob1 )
    SE2Tangentd     relative_pose; //To get the relative pose of robot 2 with respect to robot 1 
    VectorY         e1, e2;              // measurement expectation e = h(X, b1)
    MatrixT         J1_d_xi, J1_d_xj; // Jacobian of motion wrt poses_rob1 i and j
    MatrixT         J2_d_xi, J2_d_xj; // Jacobian of motion wrt poses_rob1 i and j

    MatrixT         J1_ix_x;         // Jacobian of inverse pose wrt pose
    MatrixT         J2_ix_x;         // Jacobian of inverse pose wrt pose

    MatrixYT        J1_e_ix;         // Jacobian of measurement expectation wrt inverse pose
    MatrixYT        J2_e_ix;         // Jacobian of measurement expectation wrt inverse pose

    MatrixYT        J1_e_x;          // Jacobian of measurement expectation wrt pose
    MatrixYT        J2_e_x;          // Jacobian of measurement expectation wrt pose

    MatrixYB        J1_e_b;          // Jacobian of measurement expectation wrt lmk
    MatrixYB        J2_e_b;          // Jacobian of measurement expectation wrt lmk

    SE2Tangentd     dx1, dx2;             // optimal pose correction step
    VectorB         db1, db2;             // optimal landmark correction step

    // Problem-size variables
    Matrix<double, NUM_STATES, 1>           dX1, dX2; // optimal update step for all the SAM problem
    Matrix<double, NUM_MEAS, NUM_STATES>    J1, J2;  // full Jacobian
    Matrix<double, NUM_MEAS, 1>             r1, r2;  // full residual

    /*
     * The factor graph of the SAM problem looks like this:
     *
     *                  ------- b1
     *          b3    /         |
     *          |   /       b4  |
     *          | /       /    \|
     *          X0 ---- X1 ---- X2
     *          | \   /   \   /
     *          |   b0      b2
     *          *
     *
     * where:
     *   - Xi_rob1 are poses_rob1
     *   - bk are landmarks or beacons
     *   - * is a pose prior to anchor the map and make the problem observable
     *
     * Define pairs of nodes for all the landmark measurements
     * There are 3 pose nodes [0..2] and 5 landmarks [0..4].
     * A pair pose -- lmk means that the lmk was measured from the pose
     * Each pair declares a factor in the factor graph
     * We declare 9 pairs, or 9 factors, as follows:
     */
    vector<list<int>> pairs(NUM_POSES);
    pairs[0].push_back(0);  // 0-0
    pairs[0].push_back(1);  // 0-1
    pairs[0].push_back(3);  // 0-3
    pairs[1].push_back(0);  // 1-0
    pairs[1].push_back(2);  // 1-2
    pairs[1].push_back(4);  // 1-4
    pairs[2].push_back(1);  // 2-1
    pairs[2].push_back(2);  // 2-2
    pairs[2].push_back(4);  // 2-4

    //
    //
    // END CONFIGURATION


    //// Simulator ###################################################################
    poses_simu_rob1. push_back(X_simu_rob1);
    poses_rob1.      push_back(Xi_rob1 + SE2Tangentd::Random());  // use very noisy priors

    poses_simu_rob2. push_back(X_simu_rob2);
    poses_rob2.      push_back(Xi_rob2 + SE2Tangentd::Random());  // use very noisy priors


    // temporal loop
    for (int i = 0; i < NUM_POSES; ++i)
    {
        // make measurements
        for (const auto& k : pairs[i])
        {   
            y_noise = y_sigmas * ArrayY::Random();      // measurement noise            
            // robot 1
            // simulate measurement
            b1       = landmarks_simu[k];              // lmk coordinates in world frame

            // Navid: X_simu_rob1.inverse() SE2 transformation to robot local frame
            // Navid: X_simu_rob1.inverse().act(b1) bringing landmark position to robot local frame
            // Navid: So y is how robot sees the landmark
            y1       = X_simu_rob1.inverse().act(b1);          // landmark measurement, before adding noise

            // add noise and compute prior lmk from prior pose
            measurements1[i][k]  = y1 + y_noise;           // store noisy measurements
            // Navid:  Xi_rob1 is transformation with respect to global frame
            // Navid:  Xi_rob1.act means action with respect to another transf in global fr
            b1                   = Xi_rob1.act(y1 + y_noise);   // mapped landmark with noise
            landmarks1[k]        = b1 + VectorB::Random(); // use very noisy priors

            // robot 2
            // simulate measurement
            b2       = landmarks_simu[k];              // lmk coordinates in world frame
            // same: y_noise = y_sigmas * ArrayY::Random();      // measurement noise
            // Navid: X_simu_rob1.inverse() SE2 transformation to robot local frame
            // Navid: X_simu_rob1.inverse().act(b1) bringing landmark position to robot local frame
            // Navid: So y is how robot sees the landmark
            y2       = X_simu_rob2.inverse().act(b2);          // landmark measurement, before adding noise

            // add noise and compute prior lmk from prior pose
            measurements2[i][k]  = y2 + y_noise;           // store noisy measurements
            // Navid:  Xi_rob1 is transformation with respect to global frame
            // Navid:  Xi_rob1.act means action with respect to another transf in global fr
            b2                   = Xi_rob2.act(y2 + y_noise);   // mapped landmark with noise
            landmarks2[k]        = b2 + VectorB::Random(); // use very noisy priors
        }

        // make motions
        if (i < NUM_POSES - 1) // do not make the last motion since we're done after 3rd pose
        {   
            // move simulator, without noise
            // Navid: u_nom_rob1 is an element in tangent space
            // Navid: type of u_nom_rob1 is SE2Tangentd
            // Navid: In tangent space we can add changes to state?
            X_simu_rob1 = X_simu_rob1 + u_nom_rob1;
            X_simu_rob2 = X_simu_rob2 + u_nom_rob2;
            //temp << X_simu_rob2;
            

            // move prior, with noise
            u_noise = u_sigmas * ArrayT::Random();
            Xi_rob1 = Xi_rob1 + (u_nom_rob1 + u_noise);
            Xi_rob2 = Xi_rob2 + (u_nom_rob2 + u_noise);


            // store
            poses_simu_rob1. push_back(X_simu_rob1);
            poses_simu_rob2. push_back(X_simu_rob2);
            poses_rob1.      push_back(Xi_rob1 + SE2Tangentd::Random()); // use very noisy priors
            poses_rob2.      push_back(Xi_rob2 + SE2Tangentd::Random()); // use very noisy priors
            controls1.   push_back(u_nom_rob1 + u_noise);
            controls2.   push_back(u_nom_rob2 + u_noise);
        }
    }

    //// Estimator #################################################################

    // DEBUG INFO
    cout << "prior" << std::showpos << endl;
    for (const auto& X : poses_rob1)
        cout << "pose robot 1 : " << X.translation().transpose() << " " << X.angle() << endl;
    for (const auto& X : poses_rob2)
        cout << "pose robot 2 : " << X.translation().transpose() << " " << X.angle() << endl;
    for (const auto& landmark : landmarks1)
        cout << "lmk : " << landmark.transpose() << endl;
    cout << "-----------------------------------------------" << endl;


    // iterate
    // DEBUG INFO
    cout << "iterations" << std::noshowpos << endl;
    for (int iteration = 0; iteration < MAX_ITER; ++iteration)
    {
        // Clear residual vector and Jacobian matrix
        r1 .setZero();
        J1 .setZero();

        // Navid: robot 2
        r2 .setZero();
        J2 .setZero();


        // row and column for the full Jacobian matrix J, and row for residual r
        int row = 0, col = 0;

        // 1. evaluate prior factor ---------------------
        /*
         *  NOTE (see Chapter 2, Section E, of Sola-18):
         *
         *  To compute any residual, we consider the following variables:
         *      r: residual
         *      e: expectation
         *      y: prior specification 'measurement'
         *      W: sqrt information matrix of the measurement noise.
         *
         *  In case of a non-trivial prior measurement, we need to consider
         *  the nature of it: is it a global or a local specification?
         *
         *  When prior information `y` is provided in the global reference,
         *  we need a left-minus operation (.-) to compute the residual.
         *  This is usually the case for pose priors, since it is natural
         *  to specify position and orientation wrt a global reference,
         *
         *     r = W * (e (.-) y)
         *       = W * (e * y.inv).log()
         *
         *  When `y` is provided as a local reference, then right-minus (-.) is required,
         *
         *     r = W * (e (-.) y)
         *       = W * (y.inv * e).log()
         *
         *  Notice that if y = Identity() then local and global residuals are the same.
         *
         *
         *  Here, expectation, measurement and info matrix are trivial, as follows
         *
         *  expectation
         *     e = poses_rob1[0];            // first pose
         *
         *  measurement
         *     y = SE2d::Identity()     // robot at the origin
         *
         *  info matrix:
         *     W = I                    // trivial
         *
         *  residual uses left-minus since reference measurement is global
         *     r = W * (poses_rob1[0] (.-) measurement) = log(poses_rob1[0] * Id.inv) = poses_rob1[0].log()
         *
         *  Jacobian matrix :
         *     J_r_p0 = Jr_inv(log(poses_rob1[0]))         // see proof below
         *
         *     Proof: Let p0 = poses_rob1[0] and y = measurement. We have the partials
         *       J_r_p0 = W^(T/2) * d(log(p0 * y.inv)/d(poses_rob1[0])
         *
         *     with W = i and y = I. Since d(log(r))/d(r) = Jr_inv(r) for any r in the Lie algebra, we have
         *       J_r_p0 = Jr_inv(log(p0))
         */

        // residual and Jacobian.
        // Notes:
        //   We have residual = expectation - measurement, in global tangent space
        //   We have the Jacobian in J_r_p0 = J.block<DoF, DoF>(row, col);
        // We compute the whole in a one-liner:
        // Navid: r.segment<DoF>(row) means
        // Used to extract a segment of the vector r starting from the row index and spanning DoF elements.
        r1.segment<DoF>(row)         = poses_rob1[0].lminus(SE2d::Identity(), J1.block<DoF, DoF>(row, col)).coeffs();


        d_parham=poses_simu_rob1[0].rminus(poses_simu_rob2[0]);
        tf_haniye=poses_rob2[0]*d_parham.exp();
        d1  = tf_haniye.rminus(poses_rob1[0], J1_d_xj, J1_d_xi);
        //d1  = d_parham - d_arash;
        r1.segment<DoF>(row)        +=  W_los* d1.coeffs(); 
        J1.block<DoF, DoF>(row, 0) = W_los * J1_d_xi;


        d_parham=poses_simu_rob1[1].rminus(poses_simu_rob2[1]);
        tf_haniye=poses_rob2[1]*d_parham.exp();
        d1  = tf_haniye.rminus(poses_rob1[1], J1_d_xj, J1_d_xi);
        r1.segment<DoF>(row)        +=  W_los* d1.coeffs(); 
        J1.block<DoF, DoF>(row, DoF) = W_los * J1_d_xi;

        d_parham=poses_simu_rob1[2].rminus(poses_simu_rob2[2]);
        tf_haniye=poses_rob2[2]*d_parham.exp();
        d1  = tf_haniye.rminus(poses_rob1[2], J1_d_xj, J1_d_xi);
        //d1  = poses_rob2[2].rminus(poses_rob1[2], J1_d_xj, J1_d_xi);
        r1.segment<DoF>(row)        +=  W_los* d1.coeffs(); 
        J1.block<DoF, DoF>(row, 2*DoF) = W_los * J1_d_xi;







        r2.segment<DoF>(row)         = poses_rob2[0].lminus(tf_, J2.block<DoF, DoF>(row, col)).coeffs();

        // advance rows
        row += DoF;

        // loop poses_rob1
        for (int i = 0; i < NUM_POSES; ++i)
        {
            // 2. evaluate motion factors -----------------------
            if (i < NUM_POSES - 1) // do not make the last motion since we're done after 3rd pose
            {
                int j = i + 1; // this is next pose's id

                // recover related states and data
                Xi_rob1 = poses_rob1[i];
                Xj_rob1 = poses_rob1[j];
                u1  = controls1[i];

                Xi_rob2 = poses_rob2[i];
                Xj_rob2 = poses_rob2[j];
                u2  = controls2[i];

                // expectation (use right-minus since motion measurements are local)
                /* Navid: d: This is a variable representing the expected relative motion 
                    between Xi_rob1 and Xj_rob1. 
                 J_d_xj: This is a Jacobian matrix representing the derivative of the relative motion d 
                    wrt the SE(2) pose Xj_rob1
                * Note* The method rminus() will compute this Jacobian and store the result in J_d_xj

                J_d_xi: This is a Jacobian matrix representing the derivative of the relative motion d 
                    with respect to the SE(2) pose Xi_rob1. 
                    The method rminus() will compute this Jacobian and store the result in J_d_xi.
                The rminus() method is part of the Manif library, and it is used to calculate the 
                    relative motion between two SE(2) poses_rob1. The method takes two arguments: 
                    the first pose is the "source" pose (Xi_rob1 in this case), and the second pose is 
                    the "target" pose (Xj_rob1 in this case). It returns the expected relative motion as 
                    a tangent vector in the SE(2) manifold, which is represented by the SE2Tangentd type.
                */
                d1  = Xj_rob1.rminus(Xi_rob1, J1_d_xj, J1_d_xi); // expected motion = Xj_rob1 (-) Xi_rob1
                d2  = Xj_rob2.rminus(Xi_rob2, J2_d_xj, J2_d_xi); // expected motion = Xj_rob1 (-) Xi_rob1
                
                relative_pose=Xi_rob1.rminus(Xi_rob2);
                SE2Tangentd rel_v;
                //rel_v=u1-u2;

                // residual
                r1.segment<DoF>(row)         = W * (d1 - u1).coeffs(); // residual
                r2.segment<DoF>(row)         = W * (d2 - u2).coeffs(); // residual

                // Jacobian of residual wrt first pose
                col = i * DoF;
                J1.block<DoF, DoF>(row, col) = W * J1_d_xi;
                J2.block<DoF, DoF>(row, col) = W * J2_d_xi;

                // Jacobian of residual wrt second pose
                col = j * DoF;
                J1.block<DoF, DoF>(row, col) = W * J1_d_xj;
                J2.block<DoF, DoF>(row, col) = W * J2_d_xj;

                // advance rows
                row += DoF;
            }

            // 3. evaluate measurement factors ---------------------------
            for (const auto& k : pairs[i])
            {
                // recover related states and data
                Xi_rob1 = poses_rob1[i];
                Xi_rob2 = poses_rob2[i];

                b1  = landmarks1[k];
                b2  = landmarks2[k];

                y1  = measurements1[i][k];
                y2  = measurements2[i][k];

                // expectation
                
                // EE=XX.inverse(J1).act(BB,j_EE_XX, j_EE_BB)
                // How change in XX affects inverse tf: J1
                // How change in XX affects EE: j_EE_XX
                // How change in BB affects EE: j_EE_BB

                e1       = Xi_rob1.inverse(J1_ix_x).act(b1, J1_e_ix, J1_e_b); // expected measurement = Xi_rob1.inv * bj
                J1_e_x   = J1_e_ix * J1_ix_x;                          // chain rule

                e2       = Xi_rob2.inverse(J2_ix_x).act(b2, J2_e_ix, J2_e_b); // expected measurement = Xi_rob1.inv * bj
                J2_e_x   = J2_e_ix * J2_ix_x;                          // chain rule11

                // residual
                r1.segment<Dim>(row)         = S * (e1 - y1);
                r2.segment<Dim>(row)         = S * (e2 - y2);

                // Jacobian of residual wrt pose
                col = i * DoF;
                J1.block<Dim, DoF>(row, col) = S * J1_e_x;
                J2.block<Dim, DoF>(row, col) = S * J2_e_x;


                // Jacobian of residual wrt lmk
                col = NUM_POSES * DoF + k * Dim;
                J1.block<Dim, Dim>(row, col) = S * J1_e_b;
                J2.block<Dim, Dim>(row, col) = S * J2_e_b;

                // advance rows
                row += Dim;
            }

        }

        // 4. Solve -----------------------------------------------------------------

        // compute optimal step
        // ATTENTION: This is an expensive step!!
        // ATTENTION: Use QR factorization and column reordering for larger problems!!
        dX1 = - (J1.transpose() * J1).inverse() * J1.transpose() * r1;
        dX2 = - (J2.transpose() * J2).inverse() * J2.transpose() * r2;

        // update all poses_rob1
        for (int i = 0; i < NUM_POSES; ++i)
        {
            // we go very verbose here
            int dx_row1         = i * DoF;
            constexpr int size = DoF;
            dx1                 = dX1.segment<size>(dx_row1);
            poses_rob1[i]           = poses_rob1[i] + dx1;
        }

        // update all poses_rob2
        for (int i = 0; i < NUM_POSES; ++i)
        {
            // we go very verbose here
            int dx_row2         = i * DoF;
            constexpr int size = DoF;
            dx2                 = dX2.segment<size>(dx_row2);
            poses_rob2[i]           = poses_rob2[i] + dx2;
        }


        // update all landmarks
        for (int k = 0; k < NUM_LMKS; ++k)
        {
            // we go very verbose here
            int dx_row1         = NUM_POSES * DoF + k * Dim;
            constexpr int size = Dim;
            db1                 = dX1.segment<size>(dx_row1);
            landmarks1[k]       = landmarks1[k] + db1;
        }


        // update all landmarks
        for (int k = 0; k < NUM_LMKS; ++k)
        {
            // we go very verbose here
            int dx_row2         = NUM_POSES * DoF + k * Dim;
            constexpr int size = Dim;
            db2                 = dX2.segment<size>(dx_row2);
            landmarks2[k]       = landmarks2[k] + db2;
            
        }
        
        if (iteration==0){
            cout << std :: setprecision(3) << J1 << endl;
        }
        // DEBUG INFO
        cout << "residual norm robot 1: " << std::scientific << r1.norm() << ", step norm: " << dX1.norm() << endl;
        cout << "residual norm robot 2: " << std::scientific << r2.norm() << ", step norm: " << dX2.norm() << endl;

        // conditional exit
        if (dX1.norm() < 1e-6 && dX2.norm() < 1e-6) break;
    }
    cout << "-----------------------------------------------" << endl;


    //// Print results ####################################################################

    cout << std::fixed;

    // solved problem
    cout << "posterior robot 1" << std::showpos << endl;
    for (const auto& X : poses_rob1)
        cout << "pose  : " << X.translation().transpose() << " " << X.angle() << endl;
    cout << "ground truth robot 1" << std::showpos << endl;
    for (const auto& X : poses_simu_rob1)
        cout << "pose robot 1: " << X.translation().transpose() << " " << X.angle() << endl;
    cout << "-----------------------------------------------" << endl;
    cout << "posterior robot 2" << std::showpos << endl;
    for (const auto& X : poses_rob2)
        cout << "pose  : " << X.translation().transpose() << " " << X.angle() << endl;
    cout << "ground truth robot 2" << std::showpos << endl;
    for (const auto& X : poses_simu_rob2)
        cout << "pose robot 2: " << X.translation().transpose() << " " << X.angle() << endl;
    cout << "-----------------------------------------------" << endl;
    for (const auto& landmark : landmarks1)
        cout << "lmk estimated by robot 1: " << landmark.transpose() << endl;
    cout << "-----------------------------------------------" << endl;
    for (const auto& landmark : landmarks2)
        cout << "lmk estimated by robot 2: " << landmark.transpose() << endl;
    cout << "-----------------------------------------------" << endl;
    cout << "Landmarks true position" << std::showpos << endl;
    for (const auto& landmark : landmarks_simu)
        cout << "lmk : " << landmark.transpose() << endl;
    cout << "rel pose : " << relative_pose << " " <<endl;
    cout << "-----------------------------------------------" << endl;

    // ground truth



    return 0;
}