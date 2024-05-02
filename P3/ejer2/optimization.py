import numpy as np
from scipy.optimize import minimize
from rigid import aplicar_transformacion_rigida,nifti_to_opencv

def f_rot(x,img,img_edit):
        img_edit = aplicar_transformacion_rigida(img_edit,x[0],0,0)
        diff = np.abs(img - img_edit)
        return np.sum(diff)
        # Por ejemplo, una función de prueba como la parábola en tres dimensiones

imagen_objetivo = nifti_to_opencv("images/mr1.nii",93)
imagen = aplicar_transformacion_rigida(nifti_to_opencv("images/mr1.nii",93),10,0,0)
x = np.array([0]) #[0] es rotacion [1] traslacion x [2] traslacion y

d = minimize(fun = f_rot,x0= x,args=(imagen_objetivo,imagen),method="Powell")
print("ROTACION :" + str(d.x))

def f_tras_x(x,img,img_edit):
        img_edit = aplicar_transformacion_rigida(img_edit,0,x[0],0)
        diff = np.abs(img - img_edit)
        return np.sum(diff)
        # Por ejemplo, una función de prueba como la parábola en tres dimensiones

imagen_objetivo = nifti_to_opencv("images/mr1.nii",93)
imagen = aplicar_transformacion_rigida(nifti_to_opencv("images/mr1.nii",93),0,10,0)
x = np.array([0]) #[0] es rotacion [1] traslacion x [2] traslacion y

d = minimize(fun = f_tras_x,x0= x,args=(imagen_objetivo,imagen),method="Powell")
print("TRASLACION X :" + str(d.x))


def f_tras_y(x,img,img_edit):
        img_edit = aplicar_transformacion_rigida(img_edit,0,0,x[0])
        diff = np.abs(img - img_edit)
        return np.sum(diff)
        # Por ejemplo, una función de prueba como la parábola en tres dimensiones

imagen_objetivo = nifti_to_opencv("images/mr1.nii",93)
imagen = aplicar_transformacion_rigida(nifti_to_opencv("images/mr1.nii",93),0,0,5)
x = np.array([0]) #[0] es rotacion [1] traslacion x [2] traslacion y

d = minimize(fun = f_tras_y,x0= x,args=(imagen_objetivo,imagen),method="Powell")
print("TRASLACION Y :" + str(d.x))
    