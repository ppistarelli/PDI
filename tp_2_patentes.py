import cv2
import numpy as np
import matplotlib.pyplot as plt

lista_imagenes = ["01", "02", "03", "04", "05", "08","09", "10", "11", "12"]

for nombre_imagen in lista_imagenes:
    ruta_imagen = f'img{nombre_imagen}.png'  # Formatear el nombre de la imagen
    img = cv2.imread(ruta_imagen, cv2.IMREAD_COLOR) 
    imagen_gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #plt.imshow(imagen_gris, cmap='gray'),plt.title(''),plt.show(block=False)
    ###########################----PUNTO A---##########################################################
    mediana = np.median(imagen_gris)
    q1 = np.percentile(imagen_gris, 25)
    q3 = np.percentile(imagen_gris, 75) 
    umbral_inferior = mediana - 5
    _, imagen_umbral = cv2.threshold(imagen_gris, umbral_inferior, 255, cv2.THRESH_BINARY)
    #plt.imshow(imagen_umbral, cmap='gray'), plt.title(''),plt.show(block=False)
    _, etiquetas, estadisticas, centoides = cv2.connectedComponentsWithStats(imagen_umbral)
    umbral_area = 17
    umbral_area2 = 500
    mascara_componentes = np.zeros_like(imagen_umbral)
    for i in range(1, etiquetas.max() + 1):
        area = estadisticas[i, cv2.CC_STAT_AREA]
        if area > umbral_area and area < umbral_area2:
            mascara_componentes[etiquetas == i] = 255
    #plt.imshow(mascara_componentes, cmap='gray'), plt.title(''), plt.show(block=False)
    num_labels, labels_1, stats, centroids = cv2.connectedComponentsWithStats(mascara_componentes)
    # Filtrar componentes por aspect ratio
    labels_aspect_ratio_filtered = labels_1.copy()
    for i in range(num_labels):
        if stats[i, 3] / stats[i, 2] < 1.5 or stats[i, 3] / stats[i, 2] > 3:
            labels_aspect_ratio_filtered[labels_aspect_ratio_filtered == i] = 0
    #plt.figure(), plt.imshow(labels_aspect_ratio_filtered, cmap='gray'), plt.title(''), plt.show(block=False)
    img_procesada = cv2.threshold(labels_aspect_ratio_filtered.astype(np.uint8), 1, 255, cv2.THRESH_BINARY)[1]
    #plt.figure(), plt.imshow(img_procesada, cmap='gray'), plt.title(''), plt.show(block=False)
    num_labels_casi_letras, labels_2, stats, centroids = cv2.connectedComponentsWithStats(img_procesada)
    for i in range(num_labels_casi_letras):
        contador = 0
        for j in range(num_labels_casi_letras):
            # Obtener la posición x de la componente i y j a partir de stats
            x_i = stats[i, cv2.CC_STAT_LEFT]
            x_j = stats[j, cv2.CC_STAT_LEFT]      
            y_i = stats[i, cv2.CC_STAT_TOP]
            y_j = stats[j, cv2.CC_STAT_TOP]  
            distancia_y = np.abs(y_i - y_j)
            # Calcular la distancia solo en la coordenada 
            distancia_x = np.abs(x_i - x_j)
            if distancia_x > 5 and distancia_x < 50 and distancia_y < 10:
                contador += 1
        if contador <=2:
            labels_2[labels_2 == i] = 0
    #plt.figure(), plt.imshow(labels_2, cmap='gray'),plt.title(''), plt.show(block=False)
    labels_2 = labels_2.astype(np.uint8)
    # Llamar de nuevo a componentes conectados después de filtrar labels_2
    num_labels_final, labels_final, stats_final, centroids_final = cv2.connectedComponentsWithStats(labels_2)
    indices_ordenados_por_x = np.argsort(stats_final[:, cv2.CC_STAT_LEFT])
    stats_ordenadas = stats_final[indices_ordenados_por_x]
    img_con_bounding_boxes_sorted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.figure(), plt.imshow(img_con_bounding_boxes_sorted, cmap='gray'),plt.title(''), plt.show(block=False)
    final = len(stats_ordenadas)-1
    ajuste = 23
    ajuste1 = 15
    mask = np.zeros_like(img_con_bounding_boxes_sorted)
    mask = cv2.rectangle(mask, (stats_ordenadas[1, cv2.CC_STAT_LEFT] - ajuste, stats_ordenadas[1, cv2.CC_STAT_TOP] - ajuste1), 
                                (stats_ordenadas[final, cv2.CC_STAT_LEFT] + stats_ordenadas[final, cv2.CC_STAT_WIDTH] + ajuste, stats_ordenadas[final, cv2.CC_STAT_TOP] + stats_final[final, cv2.CC_STAT_HEIGHT] + ajuste1),
                                (255, 255, 255), thickness=cv2.FILLED)
    resultado = cv2.bitwise_and(img_con_bounding_boxes_sorted, mask)
    #plt.figure(), plt.imshow(resultado), plt.show(block=False)
    img_color_modif = cv2.rectangle(img_con_bounding_boxes_sorted, (stats_ordenadas[1, cv2.CC_STAT_LEFT] - ajuste, stats_ordenadas[1, cv2.CC_STAT_TOP] - ajuste1), 
                                (stats_ordenadas[final, cv2.CC_STAT_LEFT] + stats_ordenadas[final, cv2.CC_STAT_WIDTH] + ajuste, stats_ordenadas[final, cv2.CC_STAT_TOP] + stats_final[final, cv2.CC_STAT_HEIGHT] + ajuste1),
                                (255, 0, 0), 1)
    imagen_gris = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY)
    #plt.imshow(img_color_modif, cmap='gray'),plt.title(''),plt.show(block=False)
    ###########################----PUNTO B---##########################################################
    umbral_inferior = mediana 
    _, imagen_umbral1 = cv2.threshold(imagen_gris, umbral_inferior, 255, cv2.THRESH_BINARY)
    #plt.imshow(imagen_umbral1, cmap='gray'), plt.title(''),plt.show(block=False)
    _, etiquetas, estadisticas, centoides = cv2.connectedComponentsWithStats(imagen_umbral1)
    umbral_area = 20
    umbral_area2 = 96
    mascara_componentes1 = np.zeros_like(imagen_umbral1)
    for i in range(1, etiquetas.max() + 1):
        area = estadisticas[i, cv2.CC_STAT_AREA]
        if area > umbral_area and area < umbral_area2:
            mascara_componentes1[etiquetas == i] = 255
    #plt.imshow(mascara_componentes1, cmap='gray'), plt.title(''), plt.show(block=False)
    num_labels_casi_letras1, labels_21, stats1, centroids1 = cv2.connectedComponentsWithStats(mascara_componentes1)
    #print(stats1)
    #print(num_labels_casi_letras1)
    indices_ordenados_por_y = np.argsort(stats1[:, cv2.CC_STAT_TOP])
    stats_ordenadas1 = stats1[indices_ordenados_por_y]
    #print(stats_ordenadas1)
    img_color_modificada = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for i in range(1, min(7, len(stats_ordenadas1))):
        img_color_modificada = cv2.rectangle(img_color_modificada, (stats_ordenadas1[i, cv2.CC_STAT_LEFT], stats_ordenadas1[i, cv2.CC_STAT_TOP]), (stats_ordenadas1[i, cv2.CC_STAT_LEFT]+stats_ordenadas1[i, cv2.CC_STAT_WIDTH] , stats_ordenadas1[i, cv2.CC_STAT_TOP]+stats_ordenadas1[i, cv2.CC_STAT_HEIGHT]), (255, 0, 0), 1)
    #plt.imshow(img_color_modificada, cmap='gray'), plt.title(''), plt.show(block=False)
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
    plt.figure()
    plt.suptitle(f'Imágen nro: {nombre_imagen}')
    ax1 = plt.subplot(121); imshow(img_color_modif, new_fig=False, title="Punto A", ticks=True)
    plt.subplot(122, sharex=ax1, sharey=ax1); imshow(img_color_modificada, new_fig=False, title="Punto B")
    plt.show(block=False)

input("Presiona Enter para continuar...")
plt.close('all')