import colorsys
import csv
import sys

from PIL import ImageDraw2

from pyheatmap.heatmap import HeatMap

if sys.version > "3":
    PY3 = True
else:
    PY3 = False


def mk_circle(r, w):
    u"""根据半径r以及图片宽度 w ，产生一个圆的list
    @see http://oldj.net/article/bresenham-algorithm/
    """

    # __clist = set()
    __tmp = {}

    def c8(ix, iy, v=1):
        # 8对称性
        ps = (
            (ix, iy),
            (-ix, iy),
            (ix, -iy),
            (-ix, -iy),
            (iy, ix),
            (-iy, ix),
            (iy, -ix),
            (-iy, -ix),
        )
        for x2, y2 in ps:
            p = w * y2 + x2
            __tmp.setdefault(p, v)
            # __clist.add((p, v))

    # 中点圆画法
    x = 0
    y = r
    d = 3 - (r << 1)
    while x <= y:
        for _y in range(x, y + 1):
            c8(x, _y, y + 1 - _y)
        if d < 0:
            d += (x << 2) + 6
        else:
            d += ((x - y) << 2) + 10
            y -= 1
        x += 1

    # __clist = __tmp.items()

    return __tmp.items()


def mk_colors(n=240):
    u"""生成色盘
    @see http://oldj.net/article/heat-map-colors/

    TODO: 根据 http://oldj.net/article/hsl-to-rgb/ 将 HSL 转为 RGBA
    """

    colors = []
    n1 = int(n * 0.4)
    n2 = n - n1

    for i in range(n1):
        color = "hsl(240, 100%%, %d%%)" % (100 * (n1 - i / 2) / n1)
        # color = 255 * i / n1
        colors.append(color)
    for i in range(n2):
        color = "hsl(%.0f, 100%%, 50%%)" % (240 * (1.0 - float(i) / n2))
        colors.append(color)

    return colors


class MyHeatMap(HeatMap):
    def __init__(self, **kwargs):
        super(MyHeatMap, self).__init__(**kwargs)

    def __paint_heat(self, heat_data, colors):
        u""""""

        import re

        im = self._HeatMap__im
        rr = re.compile(", (\d+)%\)")
        dr = ImageDraw2.ImageDraw.Draw(im)
        width = self.width
        height = self.height

        max_v = max(heat_data)
        if max_v <= 0:
            # 空图片
            return

        r = 240.0 / max_v
        heat_data2 = [int(i * r) - 1 for i in heat_data]

        size = width * height
        _range = range if PY3 else xrange
        for p in _range(size):
            v = heat_data2[p]
            if v > 0:
                x, y = p % width, p // width
                color = colors[v]
                alpha = int(rr.findall(color)[0])
                if alpha > 50:
                    al = 255 - 255 * (alpha - 50) // 50
                    im.putpixel((x, y), (0, 0, 255, int(0.5 * al)))
                else:
                    rgb = hsl_to_rgb(color)
                    rgba = rgb + (int(255 / 2),)
                    # print(rgb)
                    # raise Exception
                    dr.point((x, y), fill=rgba)

                    color = color[4:-2]
                    color = color.split(",")[0]
                    tmp = (x, y, int(color))
                    self.hotspot.append(tmp)

    def heatmap(self, save_as=None, base=None, data=None, r=10):
        u"""绘制热图"""

        self._HeatMap__mk_img(base)

        circle = mk_circle(r, self.width)
        heat_data = [0] * self.width * self.height

        data = data or self.data

        for hit in data:
            x, y, n = hit
            if x < 0 or x >= self.width or y < 0 or y >= self.height:
                continue

            self._HeatMap__heat(heat_data, x, y, n, circle)

        self.__paint_heat(heat_data, mk_colors())
        self._HeatMap__add_base()

        if save_as:
            self.save_as = save_as
            self._HeatMap__save()

        return self._HeatMap__im


def hsl_to_rgb(s):
    a = s[4:-1]
    hsl = [float(num.strip(" %")) for num in a.split(",")]
    hsl[1] /= 100
    hsl[2] /= 100
    r = colorsys.hls_to_rgb(hsl[0] / 360, hsl[2], hsl[1])
    return int(r[0] * 255), int(r[1] * 255), int(r[2] * 255)


s = "hsl(240, 100%, 50%)"
hsl_to_rgb(s)


def draw_heat_map(data, heatmap_name, base):
    for i, _ in enumerate(data):
        data[i][0] = round(data[i][0])
        data[i][1] = round(data[i][1])
    hm = MyHeatMap(data=data)
    # assert len(data) > 0
    # assert len(data[0]) == 2
    hm.heatmap(save_as=heatmap_name, base=base, r=40)
    return hm.hotspot


def main():

    # download test data

    data = []
    with open("/home/wtpan/memx4edu-code/exp_data/gaze.csv") as fp:
        cr = csv.reader(fp)
        next(cr)
        for row in cr:
            data.append([int(num) for num in row[1:]])

    # start painting
    hm = MyHeatMap(data=data)
    hm.clickmap(
        save_as="/home/wtpan/memx4edu-code/output/heatmap/hit_alpha.png",
        base="/home/wtpan/memx4edu-code/exp_data/WechatIMG961.png",
    )
    hm.heatmap(
        save_as="/home/wtpan/memx4edu-code/output/heatmap/heat_alpha.png",
        base="/home/wtpan/memx4edu-code/exp_data/WechatIMG961.png",
        r=40,
    )
