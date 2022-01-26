# функция для срединной точки bounding box-а лучшей предсказанной цифры на часах:
def best_detect_num(outputs_num):
    best_class = outputs_num['instances'].pred_classes.cpu().numpy()[0]
    best_class = trans_class[best_class]
    xy0 = outputs_num['instances'].pred_boxes.tensor.cpu().numpy()[0]
    x_c = 0.5 * (xy0[0] + xy0[2])
    y_c = - 0.5 * (xy0[1] + xy0[3])
    return best_class, x_c, y_c

# функция для нахождения крайних точек сегментированной стрелки:
def trr(a):
    w, h = a.shape
    coord = []
    sx = 0
    sx2 = 0
    sxy = 0
    sy= 0
    max = 9999
    min = 0
    flag = 1
    for i in range(h):
      for j in range(w):
        if a[j][i]:
          if flag:
              min = i
              flag = 0
          coord.append([i, -j])
          sx += i
          sy += -j
          sxy += i * (- j)
          sx2 += i**2
          max = i
    n = len(coord)
    a1 = (n * sxy - sx * sy) / (n * sx2 - (sx)**2)
    b1 = (sy - a1 * sx) / n
    return coord, max, min, a1, b1

# функция для длины линии по координатам:
def len_line(x1, y1, x2, y2):
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5

# вспомогательная функция:
def arr(a, b, x):
    return a * x + b

# функция для расчёта угла:
def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# дополнительно: определяем маску каждой стрелки:
def mask(out_mask, out_class):
    out_mask = outputs_arrow['instances'].pred_masks.cpu().numpy()
    out_class = outputs_arrow['instances'].pred_classes.cpu().numpy()
    if (1 in set(out_class)) and (0 in set(out_class)): # 1 - min, 0 - hour
        flagh = 1
        flagm = 1
        for i in range(out_class.shape[0]):
            if out_class[i] == 0 and flagh:
                mask_h = out_mask[i]
                flagh = 0
            elif out_class[i] == 1 and flagm:
                mask_m = out_mask[i]
                flagm = 0
    elif out_class.shape[0] >= 1:
            mask_h = out_mask[0]
            mask_m = out_mask[1]
    else:
        print('Стрелки не нашлись :(')
    return mask_h, mask_m

# определяем конец каждой стрелки:
def coord_hm(mask_m, mask_h, x0, y0):
    coord_m, max_m, min_m, a_m, b_m = trr(mask_m)
    coord_h, max_h, min_h, a_h, b_h = trr(mask_h)
    if len_line(min_h, arr(a_h, b_h, min_h), x0, y0) > len_line(max_h, arr(a_h, b_h, max_h), x0, y0):
        xh, yh = min_h, arr(a_h, b_h, min_h)
    else:
        xh, yh = max_h, arr(a_h, b_h, max_h)
    if len_line(min_m, arr(a_m, b_m, min_m), x0, y0) > len_line(max_m, arr(a_m, b_m, max_m), x0, y0):
        xm, ym = min_m, arr(a_m, b_m, min_m)
    else:
        xm, ym = max_m, arr(a_m, b_m, max_m)
    if len_line(xm, ym, x0, y0) < len_line(xh, yh, x0, y0):
        xm, ym, xh, yh = xh, yh, xm, ym
    return xh, yh, xm, ym

# определение угла от числа 12 и 3 на циферблате:
def angle_12_3(mask_m, mask_h, outputs_num):

    best_class, x_c, y_c = best_detect_num(outputs_num)
    # Координаты масок стрелок
    coord_m, max_m, min_m, a_m, b_m = trr(mask_m)
    coord_h, max_h, min_h, a_h, b_h = trr(mask_h)

    # координаты центр окружности
    x0 = (b_h - b_m) / (a_m - a_h)
    y0 = a_m * x0 + b_m

    # Координаты концов стрелок
    xh, yh, xm, ym = coord_hm(mask_m, mask_h, x0, y0)

    # Угол, на который повернуты стрелки относительно оси 0Х в противоположному часовому направлению
    angle_h = math.degrees(math.acos((xh - x0) / len_line(xh, yh, x0, y0)))
    angle_m = math.degrees(math.acos((xm - x0) / len_line(xm, ym, x0, y0)))
    if yh < y0:
        angle_h = - angle_h
    if ym < y0:
        angle_m = - angle_m

    # Координаты числа 12 и 3 на циферблате
    angle_class12 = math.degrees(math.acos((x_c - x0) / len_line(x_c, y_c, x0, y0)))
    
    if y_c > y0:
        angle_class12 = angle_class12 - (12 - best_class) * 30
    
    if angle_class12 < 0:
        angle_class12 = 360 + angle_class12
    angle_class3 = angle_class12 - 90
    angle_class12 = math.radians(angle_class12)
    angle_class3  = math.radians(angle_class3)

    x12 = x0 + math.cos(angle_class12)
    y12 = y0 + math.sin(angle_class12)
    x3 = x0 + math.cos(angle_class3)
    y3 = y0 + math.sin(angle_class3)

    # Вычисление углов часовой и минутной стрелки относительно числа 12 и 3 на циферблате
    angle_hour12 = calculate_angle((x12, y12), (x0, y0), (xh, yh))
    angle_hour3 = calculate_angle((x3, y3), (x0, y0), (xh, yh))
    angle_min12 = calculate_angle((x12, y12), (x0, y0), (xm, ym))
    angle_min3 = calculate_angle((x3, y3), (x0, y0), (xm, ym))
    return angle_hour12, angle_hour3, angle_min12, angle_min3

# функция нахождения времени:
def time(im, outputs_arrow, outputs_num):

    mask_h, mask_m = mask(outputs_arrow, outputs_num)
    
    angle_hour12, angle_hour3, angle_min12, angle_min3 = angle_12_3(mask_m, mask_h, outputs_num)

    for i in range(30):
        if i * 6 <= angle_min12 < (i + 1) * 6:
            m = i
    if angle_min3 > 90:
        m = 59 - m
          
    for i in range(6):
        if i * 30 <= angle_hour12 < (i + 1) * 30:
            h = i
    if angle_hour3 > 90:
        h = 11 - h

    hour_correct = 5    
    for i in range(6):
        if i * 30 - hour_correct <= angle_hour12 < (i + 1) * 30 - hour_correct:
            h_corr_m = i
        if i * 30 + hour_correct <= angle_hour12 < (i + 1) * 30 + hour_correct:
            h_corr_p = i

    if m <= 10:
        if angle_hour12 <= 25:
            return 0, m
        elif angle_hour12 >= 165:
            return 6, m
    elif m >= 50:
        if angle_hour12 <= 25:
            return 11, m
        elif angle_hour12 >= 165:
            return 5, m

    if m <= 10:
        if angle_hour3 < 90:
            if h_corr_m > h:
                return h_corr_m, m
        elif h_corr_p < h:
            return 11 - h_corr_p, m
    elif m >= 50:
        if angle_hour3 < 90:
            if h_corr_p < h:
                return h_corr_p, m
        elif angle_hour3 > 90:
            if h_corr_m > h:
                return 11 - h_corr_m, m


    return h, m