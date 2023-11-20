import cv2 
import numpy as np
import matplotlib.pyplot as plt

# Función para mostrar una imagen utilizando Matplotlib con opciones personalizadas.

def imshow(img, new_fig=True, title=None, color_img=False, blocking=False, colorbar=False, ticks=False):
    if new_fig:
        plt.figure()
    if color_img:
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if not ticks:
        plt.xticks([]), plt.yticks([])
    if colorbar:
        plt.colorbar()
    if new_fig:        
        plt.show(block=blocking)


#############################################-Carga de ímagen-#############################################

# Cargamos la imagen 'monedas.jpg' en formato BGR
img = cv2.imread('monedas.jpg', cv2.IMREAD_COLOR)

# Convertimos la imagen de formato BGR a RGB
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



#############################################-Tratamiento de ímagen-#############################################

# Convertimos la imagen a escala de grises
imagen_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplicamos un suavizado gaussiano a la imagen en escala de grises
# con un kernel de tamaño (5, 5) 

imagen_suavizada = cv2.GaussianBlur(imagen_gris, ksize=(5, 5), sigmaX=1.5)

'''
plt.figure()
ax1 = plt.subplot(121); imshow(img_rgb, new_fig=False, title="original", ticks=True)
plt.subplot(122, sharex=ax1, sharey=ax1); imshow(imagen_suavizada, new_fig=False, title="suavizada")
plt.show(block=False)
'''
# Detectamos bordes en la imagen suavizada usando Canny

bordes = cv2.Canny(imagen_suavizada, threshold1=20, threshold2=150)

# Creamos un elemento estructural en forma de elipse con un tamaño de (40, 40)
elemento_estructural_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (40, 40))

# Aplicamos dilatación a los bordes usando el elemento estructural
bordes_dilatados = cv2.dilate(bordes, elemento_estructural_1, iterations=1)


'''
plt.figure()
ax1 = plt.subplot(121); imshow(bordes, new_fig=False, title="Canny", ticks=True)
plt.subplot(122, sharex=ax1, sharey=ax1); imshow(bordes_dilatados, new_fig=False, title="Bordes Dilatados")
plt.show(block=False)
'''

 # Función que realiza una operación de reconstrucción morfológica
def imreconstruct(marker, mask, kernel=None):
    if kernel==None:
        kernel = np.ones((3,3), np.uint8)
    while True:
        expanded = cv2.dilate(marker, kernel)                               # Dilatacion
        expanded_intersection = cv2.bitwise_and(src1=expanded, src2=mask)   # Interseccion
        if (marker == expanded_intersection).all():                         # Finalizacion?
            break                                                           #
        marker = expanded_intersection        
    return expanded_intersection

# Función para llenar los agujeros en una imagen binaria
def imfillhole(img):
    # img: Imagen binaria de entrada. Valores permitidos: 0 (False), 255 (True).
    mask = np.zeros_like(img)                                                   # Genero mascara para...
    mask = cv2.copyMakeBorder(mask[1:-1,1:-1], 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=int(255)) # ... seleccionar los bordes.
    marker = cv2.bitwise_not(img, mask=mask)                # El marcador lo defino como el complemento de los bordes.
    img_c = cv2.bitwise_not(img)                            # La mascara la defino como el complemento de la imagen.
    img_r = imreconstruct(marker=marker, mask=img_c)        # Calculo la reconstrucción R_{f^c}(f_m)
    img_fh = cv2.bitwise_not(img_r)                         # La imagen con sus huecos rellenos es igual al complemento de la reconstruccion.
    return img_fh


img_fh = imfillhole(bordes_dilatados)


# Definimos el tamaño del kernel para la operación de erosión
L = 10
kernel_erosion = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (L,L))

# Aplicamos la operación de erosión a la imagen con agujeros rellenos
# con el kernel definido y 4 iteraciones
imagen_rellenada_erosionada = cv2.erode(img_fh, kernel_erosion, iterations=4)

'''
plt.figure()
ax1 = plt.subplot(121); imshow(img_fh, new_fig=False, title="Rellenada", ticks=True)
plt.subplot(122, sharex=ax1, sharey=ax1); imshow(imagen_rellenada_erosionada, new_fig=False, title="erosionada")
plt.show(block=False)
'''
# Definimos el tamaño del kernel para la operación de apertura
B = cv2.getStructuringElement(cv2.MORPH_RECT, (50,50))

# Aplicamos la operación de apertura a la imagen resultante de la erosión
imagen_apertura = cv2.morphologyEx(imagen_rellenada_erosionada, cv2.MORPH_OPEN, B)



#############################################-Segmentación de monedas y dados-#############################################

# obtenemos componentes conectados a la imagen después de la apertura
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(imagen_apertura)


# Factor de forma (rho)
RHO_TH = 0.83    

# Creamos una matriz de ceros con la misma forma que la matriz de labels
aux = np.zeros_like(labels)

# Creamos una imagen etiquetada con canales RGB
labeled_image = cv2.merge([aux, aux, aux])


# # Función para determinar los objetos circulares y no circulares basándose en el factor de forma "rho".
for i in range(1, num_labels):
    # --- Selecciono el objeto actual -----------------------------------------
    obj = (labels == i).astype(np.uint8)
    # --- Calculo Rho ---------------------------------------------------------
    ext_contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(ext_contours[0])
    perimeter = cv2.arcLength(ext_contours[0], True)
    rho = 4 * np.pi * area/(perimeter**2)
    flag_circular = rho > RHO_TH
    if flag_circular:
            labeled_image[obj == 1, 0] = 255   # Circular
       
    else:
        labeled_image[obj == 1, 2] = 255        # No circular

'''
plt.figure()
ax1 = plt.subplot(); imshow(labeled_image, new_fig=False, title="Coloreado de tipo de objetos", ticks=True)
plt.show(block=False)
'''
#############################################-Reconocimiento de monedas y dados-#############################################

img_color_modificada = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cant_monedas = 0
cant_dados = 0
monedas_1= 0
monedas_50= 0
monedas_10= 0
lista_dados = []
for i in range(1, num_labels):
    # --- Selecciono el objeto actual -----------------------------------------
    obj = (labels == i).astype(np.uint8)
    # --- Calculo Rho ---------------------------------------------------------
    ext_contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(ext_contours[0])
    perimeter = cv2.arcLength(ext_contours[0], True)
    rho = 4 * np.pi * area/(perimeter**2)
    
    # Verificamos si el objeto es circular 
    flag_circular = rho > RHO_TH
    if flag_circular:
        cant_monedas +=1 
        # Determinamos el tipo de moneda (1, 10, 50) según el área (elegida mediante pruebas) y dibujamos un rectangulo de color sobre cada una en la imagen
        if stats[i, cv2.CC_STAT_AREA] < 80000:
            img_color_modificada = cv2.rectangle(img_color_modificada, (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]), (stats[i, cv2.CC_STAT_LEFT]+stats[i, cv2.CC_STAT_WIDTH] , stats[i, cv2.CC_STAT_TOP]+stats[i, cv2.CC_STAT_HEIGHT]), (255, 0, 0), 4)        
            monedas_10+=1 
        elif stats[i, cv2.CC_STAT_AREA] > 90000:
            img_color_modificada = cv2.rectangle(img_color_modificada, (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]), (stats[i, cv2.CC_STAT_LEFT]+stats[i, cv2.CC_STAT_WIDTH] , stats[i, cv2.CC_STAT_TOP]+stats[i, cv2.CC_STAT_HEIGHT]), (0, 255, 0), 4)        
            monedas_50+=1  
        else:            
            img_color_modificada = cv2.rectangle(img_color_modificada, (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]), (stats[i, cv2.CC_STAT_LEFT]+stats[i, cv2.CC_STAT_WIDTH] , stats[i, cv2.CC_STAT_TOP]+stats[i, cv2.CC_STAT_HEIGHT]), (0, 255, 255), 4)        
            monedas_1+=1 
    else: # Objeto es un dado
        cant_dados+=1
        ajuste_dado = 25
        mask = np.zeros_like(img_color_modificada)
        # Dibujamos un rectangulo de color sobre cada dado en la imagen
        img_color_modificada = cv2.rectangle(img_color_modificada, (stats[i, cv2.CC_STAT_LEFT]-ajuste_dado, stats[i, cv2.CC_STAT_TOP]-ajuste_dado), (stats[i, cv2.CC_STAT_LEFT]+stats[i, cv2.CC_STAT_WIDTH]+ajuste_dado , stats[i, cv2.CC_STAT_TOP]+stats[i, cv2.CC_STAT_HEIGHT]+ajuste_dado), (0, 0, 255), 4)
        # Creamos una máscara para aislar el dado en la imagen
        mask = cv2.rectangle(mask, (stats[i, cv2.CC_STAT_LEFT] - ajuste_dado, stats[i, cv2.CC_STAT_TOP] - ajuste_dado), 
                            (stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] + ajuste_dado, stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT] + ajuste_dado),
                            (255, 255, 255), thickness=cv2.FILLED)
        resultado = cv2.bitwise_and(img_color_modificada, mask)
        # Recortamos la región del dado y la convertirmos a escala de grises
        mask_grayscale = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        mask_uint8 = mask_grayscale.astype(np.uint8)
        x, y, w, h = cv2.boundingRect(mask_uint8)
        resultado_recortado = resultado[y:y+h, x:x+w]
        imagen_gris = cv2.cvtColor(resultado_recortado, cv2.COLOR_BGR2GRAY)
        # Detectamos y contamos los puntos en cada dado mediante el cálculo del Rho
        f_point = imagen_gris < 100
        f_point = f_point.astype(np.uint8)
        elemento_estructural_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dado_dilatado = cv2.dilate(f_point, elemento_estructural_3, iterations=1) 
        num_labels_dados, labels_dados, stats_dados, centroids_dados = cv2.connectedComponentsWithStats(dado_dilatado, 8, cv2.CV_32S)
        contador_numeros_dados=0
        for i in range(1, num_labels_dados):
            obj = (labels_dados == i).astype(np.uint8)
            ext_contours, _ = cv2.findContours(obj, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            area_dados = cv2.contourArea(ext_contours[0])
            perimeter_dados = cv2.arcLength(ext_contours[0], True)
            rho = 4 * np.pi * area_dados/(perimeter_dados**2)
            flag_circular = rho > RHO_TH
            if flag_circular:
                    contador_numeros_dados += 1  
        lista_dados.append(contador_numeros_dados)    
            
        
#############################################-Generación de ímagen final con datos de monedas y dados-#############################################

# Factor de escala vertical para agrandar la imagen
escala_vertical = 1.5

# Obtenemos dimensiones originales de la imagen modificada
alto_original, ancho_original = img_color_modificada.shape[:2]

# Calculamos nuevas dimensiones para la imagen agrandada
alto_nuevo = int(alto_original * escala_vertical)
ancho_nuevo = ancho_original

# Creamos una imagen agrandada con fondo gris
imagen_agrandada = np.ones((alto_nuevo, ancho_nuevo, 3), dtype=np.uint8) * 200

# Copiamos la imagen modificada a la parte superior de la imagen agrandada
imagen_agrandada[:alto_original, :] = img_color_modificada

# Agregamos texto con la información a mostrar de monedas y dados a la imagen agrandada
imagen_agrandada = cv2.putText(imagen_agrandada, f'Cantidad total de monedas: {cant_monedas}', (400,3000), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 8)
imagen_agrandada = cv2.putText(imagen_agrandada, f'Cantidad de monedas de 10 centavos: {monedas_10}', (400,3150), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 0, 0), 8)
imagen_agrandada = cv2.putText(imagen_agrandada, f'Cantidad de monedas de 1 peso: {monedas_1}', (400,3300), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 255), 8)
imagen_agrandada = cv2.putText(imagen_agrandada, f'Cantidad de monedas de 50 centavos: {monedas_50}', (400,3450), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 255, 0), 8)
imagen_agrandada = cv2.putText(imagen_agrandada, f'Cantidad de dados: {cant_dados}', (2500,3000), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 8)
imagen_agrandada = cv2.putText(imagen_agrandada, f'Numeros de dado 1: {lista_dados[0]}', (2500,3150), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 8)
imagen_agrandada = cv2.putText(imagen_agrandada, f'Numeros de dado 2: {lista_dados[1]}', (2500,3300), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (0, 0, 255), 8)


plt.figure(), plt.imshow(imagen_agrandada), plt.show(block=False)











