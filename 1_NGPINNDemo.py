import warp as wp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

MODE = "HMO"

#---------------------------------------------------------------------------------------------
# 0] Constants Physics Equation that we want to solve
MAX_T = 5.0 
MAX_M = 10.0
g     = 9.81
MAX_S = 0.5 * g * (MAX_T**2)
MAX_V = g * MAX_T
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# HMO Constants
M_VAL = 1.0
C_VAL = 1.0  # Damping
K_VAL = 2.0  # Spring Stiffness (High K = Fast oscillation)
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# 0.1] Details of the MultiLayer NN 
m_numInputs  = 2   
m_numLayers  = 4    # Increase depth to capture curvature each layer increases order
m_numNeurons = 128  # Increase width for better expressiveness if correlations are more complex 
m_numOutputs = 2    #  Pos Vel

m_epochs    = 50000  # Physics needs time to converge
m_learnRate = 0.001  # Slower, steadier learning rate
#---------------------------------------------------------------------------------------------


#---------------------------------------------------------------------------------------------
# 1] Wakes up the driver, from a slow python to a Cobra ;-) ( really I suggested this instead of warp) 
wp.init()
#wp.config.verify_cuda = True
SetDevice = "cuda"
#---------------------------------------------------------------------------------------------

#=============================================================================================
# Kernels Get JIT into CUDA ( Python people should learn how to code ;-) )
#=============================================================================================

#---------------------------------------------------------------------------------------------
# 2.0] Neuron Activation (Domain Mapping so we can handle derivs) : cuda inline device function 
@wp.func
def swish(x: float):
    # Swish (x * sigmoid(x)) is better for physics derivatives than Tanh
    return x / (1.0 + wp.exp(-x))
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# 2.0b] cudaMemset
@wp.kernel
def zero_scalar_kernel(scalar_arr: wp.array(dtype=float)):
    # Explicitly reset the accumulator to 0.0 on the GPU
    scalar_arr[0] = float(0.0)
#---------------------------------------------------------------------------------------------

##---------------------------------------------------------------------------------------------
# 2.1a] Loss Function using a "equation" for "truth" computed as least square error (DATA MODE)
##---------------------------------------------------------------------------------------------
@wp.kernel
def CUDAKern_Loss_DD(
    OutputVector:      wp.array(dtype=float), 
    OutputVectorTruth: wp.array(dtype=float), 
    LossVector:        wp.array(dtype=float), 
    lossTot:           wp.array(dtype=float), 
    numOutput:         int
):
    tid = wp.tid()
    if tid < numOutput:
        # Calculate individual error (Residual)
        diff = OutputVector[tid] - OutputVectorTruth[tid]
        LossVector[tid] = diff
        
        # Atomically add the squared error to the total sum
        wp.atomic_add(lossTot, 0, diff * diff)
##---------------------------------------------------------------------------------------------

##---------------------------------------------------------------------------------------------
# 2.1b] PINN Loss Function: "I don't have data, I only have the Physics Law S = 0.5*a*t^2"
##---------------------------------------------------------------------------------------------
@wp.kernel
def CUDAKern_Loss_PINN(
    OutputVector: wp.array(dtype=float),    # Network Prediction [s, v]
    InputVector:  wp.array(dtype=float),    # Network Input [t, m]
    lossTot:      wp.array(dtype=float),    # Scalar accumulator
    g: float, max_s: float, max_v: float, max_t: float # defines the equation we are solving
):
    # We only need one thread to check the physics for this sample
    tid = wp.tid()
    if tid == 0:
        # 1. Denormalize the Input Time to get real seconds
        t = InputVector[0] * max_t
        
        # 2. Compute the Physics Laws on the Fly (The Equation)
        s_physics = (0.5 * g * t * t) / max_s  # Normalized S
        v_physics = (g * t) / max_v            # Normalized V
        
        # 3. Compute Residuals (Network Prediction - Physics Law)
        pred_s = OutputVector[0]
        pred_v = OutputVector[1]
        
        diff_s = pred_s - s_physics
        diff_v = pred_v - v_physics
        
        # 4. Total PINN Loss
        total_error = (diff_s * diff_s) + (diff_v * diff_v)
        
        wp.atomic_add(lossTot, 0, total_error)
##---------------------------------------------------------------------------------------------

##---------------------------------------------------------------------------------------------
# 2.1c] HMO Loss Function: Damped Harmonic Oscillator (Mass-Spring-Damper)
# System: m*a + c*v + k*x = 0  |  Initial: x(0)=1, v(0)=0
##---------------------------------------------------------------------------------------------
@wp.kernel
def CUDAKern_Loss_HMO(
    OutputVector: wp.array(dtype=float),    # Network Prediction [x, v]
    InputVector:  wp.array(dtype=float),    # Network Input [t, m]
    lossTot:      wp.array(dtype=float),    # Scalar accumulator
    max_t: float,
    # Physical Constants
    m: float, c: float, k: float
):
    tid = wp.tid()
    if tid == 0:
        # 1. Denormalize Time
        t = InputVector[0] * max_t
        
        # 2. Calculate Physics Constants
        # Natural Frequency (wn) and Damping Ratio (zeta)
        wn = wp.sqrt(k / m)
        zeta = c / (2.0 * wp.sqrt(m * k))
        
        # Damped Frequency (wd)
        sqrt_1_z2 = wp.sqrt(1.0 - zeta*zeta)
        wd = wn * sqrt_1_z2
        
        # 3. Compute Analytical Truth (Underdamped Case)
        decay = wp.exp(-zeta * wn * t)
        sin_val = wp.sin(wd * t)
        cos_val = wp.cos(wd * t)
        
        # Position x(t)
        x_true = decay * (cos_val + (zeta / sqrt_1_z2) * sin_val)
        
        # Velocity v(t) - Derivative of x(t)
        v_true = - (wn / sqrt_1_z2) * decay * sin_val
        
        # 4. Compute Residuals
        # Note: We assume Network Output is ALREADY normalized to roughly [-1, 1] range
        # Since HMO amplitude is max 1.0, we don't need huge scaling factors here
        pred_x = OutputVector[0]
        pred_v = OutputVector[1]
        
        diff_x = pred_x - x_true
        diff_v = pred_v - v_true # You might want to scale this if v is large
        
        total_error = (diff_x * diff_x) + (diff_v * diff_v)
        
        wp.atomic_add(lossTot, 0, total_error)


##---------------------------------------------------------------------------------------------
# 2.2]  SGD Update Kernel
@wp.kernel
def CUDAKern_SGDUpdate(w: wp.array(dtype=float), g: wp.array(dtype=float), lr: float):
    tid = wp.tid()
    w[tid] = w[tid] - lr * g[tid]
##---------------------------------------------------------------------------------------------    


#---------------------------------------------------------------------------------------------
# 3.1] Input Layer: Input -> Value Vector  
#  WeightMatrix: [numNeurons][numOutput], InputVector: [numInputs], NeuronValVector: [numNeurons]  
#---------------------------------------------------------------------------------------------
@wp.kernel
def CUDAKern_NN_InputLayer(
    InputVector:     wp.array(dtype=float),
    WeightMatrix:    wp.array(dtype=float, ndim=3), # layer, inputRow, neuron
    BiasVector:      wp.array(dtype=float, ndim=2), 
    NeuronValVector: wp.array(dtype=float, ndim=2), # layer, neuron 
    numInput: int, numOutput: int
):
    indexNeuron = wp.tid()
    # 1] For each thread we sum the product of its Weight row with the input row vector 
    if indexNeuron < numOutput:
        dot = float(0.0)
        for indexRow in range(numInput):
            dot += WeightMatrix[0, indexRow, indexNeuron] * InputVector[indexRow]

        # 2] NeuronValue: Activation Function to Map from "world" to "nice" domain 
        NeuronValVector[0, indexNeuron] = swish(dot + BiasVector[0, indexNeuron])
#---------------------------------------------------------------------------------------------

#---------------------------------------------------------------------------------------------
# 3.2] Hidden Layers
# NeuronValVector [numNeurons] -> ValueMatrix [numNeurons] , WeightMatrix: [numNeurons][numOutput],  
#---------------------------------------------------------------------------------------------
@wp.kernel
def CUDAKern_NN_HiddenLayers(
    NeuronValVector: wp.array(dtype=float, ndim=2), # layer -1 is our input
    WeightMatrix:    wp.array(dtype=float, ndim=3), 
    BiasVector:      wp.array(dtype=float, ndim=2), 
    indexLayer: int, numNeuron: int      
):
    indexNeuron = wp.tid()
    # 1] For each thread we sum the product of its Weight row with the Value Row 
    if indexNeuron < numNeuron:
        dot = float(0.0)
        for indexCol in range(numNeuron): # Loop over numNeurons to do the row col sum 
            dot += WeightMatrix[indexLayer, indexCol, indexNeuron] * NeuronValVector[indexLayer - 1, indexCol]
        
        # 2] NeuronValue: Activation Function to Map from "world" to "nice" domain 
        NeuronValVector[indexLayer, indexNeuron] = swish(dot + BiasVector[indexLayer, indexNeuron])
#---------------------------------------------------------------------------------------------


##---------------------------------------------------------------------------------------------
# 3.3] Output Layer:
# NeuronValVector [numNeurons] -> OutputVector [numOutputs] , WeightMatrix: [numOutputs][numNeurons]  
#---------------------------------------------------------------------------------------------
@wp.kernel
def CUDAKern_NN_OutputLayer(
    NeuronValVector: wp.array(dtype=float, ndim=2),
    WeightMatrix:    wp.array(dtype=float, ndim=3),
    BiasVector:      wp.array(dtype=float, ndim=2), 
    OutputVector:    wp.array(dtype=float), 
    indexLayer: int, numInput: int, numOutput: int
):
    indexOutput = wp.tid()
     # 1] For each thread which is the output we sum the product of its Weight row with the Value Col 
    if indexOutput < numOutput:
        dot = float(0.0)
        for j in range(numInput):
            dot += WeightMatrix[indexLayer, j, indexOutput] * NeuronValVector[indexLayer-1, j]
        OutputVector[indexOutput] = dot + BiasVector[indexLayer, indexOutput]
##---------------------------------------------------------------------------------------------

#=============================================================================================
# End CUDA Kernels, Now we slow again with python 
#=============================================================================================


#---------------------------------------------------------------------------------------------
# Init GPU Arrays and Diff
#---------------------------------------------------------------------------------------------
def InitGPUMemory(numLayers, numInputs, numNeurons, numOutputs, device="cuda" ): # GPU Memory  

    max_dim = max(numInputs, numNeurons, numOutputs)
    W_np    = np.zeros((numLayers, max_dim, max_dim), dtype=np.float32)
    
    #-------------------------
    # Host Init 
    # We use a larger factor (2.0) to encourage activity
    s0 = np.sqrt(4.0/numInputs) 
    W_np[0,:numInputs,:numNeurons] = np.random.normal(0, s0, (numInputs, numNeurons))
    
    sh = np.sqrt(4.0/numNeurons)
    for l in range(1, numLayers-1): 
        W_np[l,:numNeurons,:numNeurons] = np.random.normal(0, sh, (numNeurons, numNeurons))
        
    so = np.sqrt(4.0/numNeurons)
    W_np[numLayers-1,:numNeurons,:numOutputs] = np.random.normal(0, so, (numNeurons, numOutputs))
    #-------------------------

    #-------------------------
    # Copy to GPU
    W = wp.from_numpy(W_np, dtype=wp.float32, device=device, requires_grad=True)             # Set Weight and tell cuda we want the instructions "backward" 
    B = wp.zeros((numLayers, max_dim), dtype=wp.float32, device=device, requires_grad=True)  # Set Bias and tell cuda we want the instructions "backward"
    V = wp.zeros((numLayers, max_dim), dtype=wp.float32, device=device, requires_grad=True)  # Set Values and tell cuda we want the instructions "backward"
    #-------------------------
    
    return W, B, V  # Return GPU Arrays
#---------------------------------------------------------------------------------------------



#---------------------------------------------------------------------------------------------
# Main Loop 
#---------------------------------------------------------------------------------------------
def main():
    print(f"Running PINN Warp Training on {SetDevice}...")
    
    #------------------------------------------------------
    USE_PINN_MODE = True  
    # False = Use Data Loss (Curve Fitting)
    # True  = Use PINN Loss (Physics Equation S=0.5at^2)

    if USE_PINN_MODE:
        print(">> MODE: PINN (Learning purely from the Equation)")
    else:
        print(">> MODE: Curve Fitting (Learning from Data)")
    #------------------------------------------------------

    #------------------------------------------------------
    # CudaMalloc and Init
    W, B, V = InitGPUMemory(m_numLayers, m_numInputs, m_numNeurons, m_numOutputs, device=SetDevice)
    
    CudaWP_InputVec    = wp.zeros(m_numInputs, dtype=float, device=SetDevice)
    CudaWP_OutVecTruth = wp.zeros(m_numOutputs, dtype=float, device=SetDevice) # Needed for Data Mode
    CudaWP_OutVecPred  = wp.zeros(m_numOutputs, dtype=float, device=SetDevice, requires_grad=True)
    CudaWP_LossVec     = wp.zeros(m_numOutputs, dtype=float, device=SetDevice) 
    CudaWP_LossSum     = wp.zeros(1, dtype=float, device=SetDevice, requires_grad=True) 
    #------------------------------------------------------


    print("⏳ Starting Training Timer")
    wp.synchronize() 
    start_time = time.perf_counter()
    #------------------------------------------------------
    # Start Training
    #------------------------------------------------------
    for epoch in range(m_epochs):

        # Each step we generate a random value for the input parameters
        t_raw = np.random.uniform(0.0, MAX_T)
        m_raw = np.random.uniform(1.0, MAX_M)
        
        # Scale so the range is nicer, also a reason AI people killed FP32/64
        t_norm = t_raw / MAX_T
        m_norm = m_raw / MAX_M
        
        # 0.1] Copy the new input vector to GPU
        wp.copy(CudaWP_InputVec, wp.from_numpy(np.array([t_norm, m_norm], dtype=np.float32), device=SetDevice))


        #------------------------------------------------------
        # 0.2] If Data Mode, we must calculate the "Truth" and send it to GPU
        if not USE_PINN_MODE:
            # We use the current MODE variable to decide what "Truth" data to generate
            if MODE == "GRAVITY":
                s_target_norm = (0.5 * g * t_raw**2) / MAX_S
                v_target_norm = (g * t_raw) / MAX_V
            elif MODE == "HMO":
                # Add HMO Data Gen here if you want to curve fit HMO data
                wn = np.sqrt(K_VAL/M_VAL)
                zeta = C_VAL / (2*np.sqrt(M_VAL*K_VAL))
                wd = wn * np.sqrt(1 - zeta**2)
                decay = np.exp(-zeta * wn * t_raw)
                s_target_norm = decay * (np.cos(wd * t_raw) + (zeta / np.sqrt(1 - zeta**2)) * np.sin(wd * t_raw))
                # For simplicity in data fitting, we can just fit Position (index 0) or both
                v_target_norm = 0.0 # Placeholder if lazy, or calc real derivative

            wp.copy(CudaWP_OutVecTruth, wp.from_numpy(np.array([s_target_norm, v_target_norm], dtype=np.float32), device=SetDevice))
        #------------------------------------------------------

        
        #------------------------------------------------------
        tape = wp.Tape() # records instructions 
        with tape:
            # 1] Set the loss to zero 
            wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[CudaWP_LossSum], device=SetDevice)

            # 2] Launch the Input Layer
            wp.launch(CUDAKern_NN_InputLayer, dim=m_numNeurons, inputs=[CudaWP_InputVec, W, B, V, m_numInputs, m_numNeurons], device=SetDevice)
            
            # 3] Set the hidden layer cant do async since each layer needs the previous so only paralllel in numNeurons
            for l in range(1, m_numLayers - 1):
                wp.launch(CUDAKern_NN_HiddenLayers, dim=m_numNeurons, inputs=[V, W, B, l, m_numNeurons], device=SetDevice)

            # 4] Launch Output Layer   
            wp.launch(CUDAKern_NN_OutputLayer, dim=m_numOutputs, inputs=[V, W, B, CudaWP_OutVecPred, m_numLayers-1, m_numNeurons, m_numOutputs], device=SetDevice)
            
            # 5] Magic Choose Loss Function
            if USE_PINN_MODE:
                if MODE == "HMO":
                    wp.launch(CUDAKern_Loss_HMO, dim=1,
                            inputs=[CudaWP_OutVecPred, CudaWP_InputVec, CudaWP_LossSum, MAX_T, M_VAL, C_VAL, K_VAL],
                            device=SetDevice)
                elif MODE == "GRAVITY":
                    wp.launch(CUDAKern_Loss_PINN, dim=1, 
                            inputs=[CudaWP_OutVecPred, CudaWP_InputVec, CudaWP_LossSum, g, MAX_S, MAX_V, MAX_T], 
                            device=SetDevice)
            else: # DATA FITTING
                wp.launch(CUDAKern_Loss_DD, dim=m_numOutputs, 
                        inputs=[CudaWP_OutVecPred, CudaWP_OutVecTruth, CudaWP_LossVec, CudaWP_LossSum, m_numOutputs], 
                        device=SetDevice)
        #------------------------------------------------------

        #------------------------------------------------------
        # 6] More Magic does the back propgation of the derative functions by replaying instructions updating weights
        tape.backward(loss=CudaWP_LossSum)
         #------------------------------------------------------

        #------------------------------------------------------
        # 7] Stocastic Gradient Descent, this is where the "learning' happens the neurons that contribute the most to the error get a high gradient so the weight goes down
        wp.launch(CUDAKern_SGDUpdate, dim=W.size, inputs=[W.flatten(), W.grad.flatten(), m_learnRate], device=SetDevice)
        wp.launch(CUDAKern_SGDUpdate, dim=B.size, inputs=[B.flatten(), B.grad.flatten(), m_learnRate], device=SetDevice)
        #------------------------------------------------------

        # Clear the tape for the next run 
        tape.zero()

        if epoch % 5000 == 0:
            print(f"Epoch {epoch} | Loss: {CudaWP_LossSum.numpy()[0]:.6f}")
    #------------------------------------------------------
    # End Training
    #------------------------------------------------------
    wp.synchronize() # Wait for everything to finish
    end_time = time.perf_counter()
    total_time = end_time - start_time

    #------------------------------------------------------
    # Results
    #------------------------------------------------------
    print("-" * 60)
    print(f"✅ TRAINING COMPLETE in {total_time:.2f}s ({m_epochs/total_time:.0f} epochs/s)")
    
    t_vals = np.linspace(0, MAX_T, 100)
    s_p, v_p = [], []
    
    #------------------------------------------------------
    # Generate Analytical Truth for Comparison
    if MODE == "GRAVITY":
        truth_s = 0.5 * g * t_vals**2
        truth_v = g * t_vals
        lbl_s, lbl_v = "Position (m)", "Velocity (m/s)"
    elif MODE == "HMO":
        wn = np.sqrt(K_VAL/M_VAL)
        zeta = C_VAL / (2*np.sqrt(M_VAL*K_VAL))
        wd = wn * np.sqrt(1 - zeta**2)
        decay = np.exp(-zeta * wn * t_vals)
        truth_s = decay * (np.cos(wd * t_vals) + (zeta / np.sqrt(1 - zeta**2)) * np.sin(wd * t_vals))
        truth_v = - (wn / np.sqrt(1 - zeta**2)) * decay * np.sin(wd * t_vals)
        lbl_s, lbl_v = "Displacement (x)", "Velocity (v)"
    #------------------------------------------------------    

    #------------------------------------------------------
    # Inference Loop
    #------------------------------------------------------
    for t in t_vals:
        wp.copy(CudaWP_InputVec, wp.from_numpy(np.array([t/MAX_T, 0.5], dtype=np.float32), device=SetDevice))
        wp.launch(CUDAKern_NN_InputLayer, dim=m_numNeurons, inputs=[CudaWP_InputVec, W, B, V, m_numInputs, m_numNeurons], device=SetDevice)
        for l in range(1, m_numLayers - 1):
            wp.launch(CUDAKern_NN_HiddenLayers, dim=m_numNeurons, inputs=[V, W, B, l, m_numNeurons], device=SetDevice)
        wp.launch(CUDAKern_NN_OutputLayer, dim=m_numOutputs, inputs=[V, W, B, CudaWP_OutVecPred, m_numLayers-1, m_numNeurons, m_numOutputs], device=SetDevice)
        
        res = CudaWP_OutVecPred.numpy()
        # Scale back up (Gravity uses scale, HMO is already approx 1.0)
        scale_s = MAX_S if MODE == "GRAVITY" else 1.0
        scale_v = MAX_V if MODE == "GRAVITY" else 1.0
        
        s_p.append(res[0] * scale_s)
        v_p.append(res[1] * scale_v)
    #------------------------------------------------------


    # Accuracy Calc
    acc = (1.0 - np.linalg.norm(s_p - truth_s)/np.linalg.norm(truth_s)) * 100.0
    print(f"🎯 Final Accuracy: {acc:.2f}%")

    plt.figure(figsize=(10, 5))
    
    # Subplot 1: Position
    plt.subplot(1, 2, 1)
    plt.plot(t_vals, truth_s, 'r--', label='Physics Truth', linewidth=2)
    plt.plot(t_vals, s_p, 'b-', label=f'AI Prediction (isPINN_{USE_PINN_MODE})', linewidth=1.5)
    plt.title(f"{MODE}: {lbl_s}\nAccuracy: {acc:.2f}%")
    plt.xlabel("Time (s)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Subplot 2: Velocity
    plt.subplot(1, 2, 2)
    plt.plot(t_vals, truth_v, 'r--', label='Physics Truth', linewidth=2)
    plt.plot(t_vals, v_p, 'g-', label=f'AI Prediction (isPINN_{USE_PINN_MODE})', linewidth=1.5)
    plt.title(f"{MODE}: {lbl_v}")
    plt.xlabel("Time (s)")
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save
    filename = f"Result_isPINN_{USE_PINN_MODE}_Case_{MODE}.png"
    plt.tight_layout()
    plt.savefig(filename)
    print(f"📊 Graph saved to: {filename}")

if __name__ == "__main__":
    main()