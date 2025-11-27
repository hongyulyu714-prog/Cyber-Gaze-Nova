import cv2 as cv
import mediapipe as mp
import pygame as pg
import numpy as np
import random
import math
winWidth = 1280
winHeight = 720
maxOrbs = 1500
burstForce = 25.0
sensitivity = 0.6
smoothFactor = 0.05
# Colors
colorPink = (255, 105, 180)
colorCyan = (0, 255, 240)
colorYellow = (255, 255, 0)
colorWhite = (255, 255, 255)
colorBlackTrans = (0, 0, 0, 180)
colorGreen = (50, 255, 50)
colorRed = (255, 50, 50)
orbList = []
pg.init()
mainCanvas = pg.display.set_mode((winWidth, winHeight))
pg.display.set_caption("Eye Control System // Eye Tracker HUD")
appClock = pg.time.Clock()
mpFaceMesh = mp.solutions.face_mesh
faceTracker = mpFaceMesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

camSource = cv.VideoCapture(0)
camSource.set(cv.CAP_PROP_FRAME_WIDTH, winWidth)
camSource.set(cv.CAP_PROP_FRAME_HEIGHT, winHeight)


# --- 3. Font System ---
def getSafeFont(size, isBold=False):
    macFonts = ["pingfangsc", "pingfangtc", "heititc", "stheiti", "songtisc", "arialunicode"]
    font = pg.font.SysFont(macFonts, size, bold=isBold)
    return font


fontTitleBig = getSafeFont(28, True)
fontTitleSub = getSafeFont(16)
fontStatus = getSafeFont(20, True)
fontBody = getSafeFont(15)


# --- 4. Particle Class ---
class CyberOrb:
    def __init__(self):
        self.pos = pg.Vector2(random.randint(0, winWidth), random.randint(0, winHeight))
        self.vel = pg.Vector2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
        if random.random() < 0.85:
            self.baseSize = random.randint(1, 3)
        else:
            self.baseSize = random.randint(4, 6)
        self.currentSize = self.baseSize
        self.color = random.choice([colorPink, colorCyan, colorYellow])
        self.pulseOffset = random.random() * 6.28

    def applyImpulse(self, centerPos, force):
        direction = self.pos - centerPos
        if direction.length() == 0:
            direction = pg.Vector2(random.uniform(-1, 1), random.uniform(-1, 1))
        direction.normalize_ip()
        self.vel += direction * force * random.uniform(0.5, 1.5)

    def update(self, targetPos, isGathering):
        if isGathering:
            direction = targetPos - self.pos
            dist = direction.length()
            if dist > 10:
                direction.normalize_ip()
                pullStrength = min(15.0, dist / 20.0)
                self.vel += direction * pullStrength
            self.vel *= 0.80
            self.currentSize = self.baseSize * 1.8
        else:
            self.vel *= 0.94
            self.currentSize = self.baseSize
            if self.pos.x < 0: self.pos.x = winWidth
            if self.pos.x > winWidth: self.pos.x = 0
            if self.pos.y < 0: self.pos.y = winHeight
            if self.pos.y > winHeight: self.pos.y = 0
        self.pos += self.vel

    def draw(self, surface, timeTicks):
        pulse = math.sin(timeTicks * 0.003 + self.pulseOffset)
        if self.baseSize > 3:
            drawSize = self.currentSize * (0.8 + pulse * 0.2)
            alpha = int(180 + pulse * 50)
            orbSurf = pg.Surface((int(drawSize * 4), int(drawSize * 4)), pg.SRCALPHA)
            pg.draw.circle(orbSurf, (*colorWhite, alpha), (int(drawSize * 2), int(drawSize * 2)), int(drawSize / 2))
            pg.draw.circle(orbSurf, (*self.color, alpha // 2), (int(drawSize * 2), int(drawSize * 2)), int(drawSize))
            surface.blit(orbSurf, (self.pos.x - drawSize * 2, self.pos.y - drawSize * 2), special_flags=pg.BLEND_ADD)
        else:
            alpha = int(120 + pulse * 80)
            pg.draw.circle(surface, (*self.color, alpha), (int(self.pos.x), int(self.pos.y)), self.baseSize)


class GazeAndBlinkSystem:
    def __init__(self):
        self.currentX = winWidth / 2
        self.currentY = winHeight / 2
        self.eyesClosed = False

    def getEyeOpenness(self, landmarks):
        lTop = landmarks[159]
        lBot = landmarks[145]
        rTop = landmarks[386]
        rBot = landmarks[374]
        avgDist = (math.dist((lTop.x, lTop.y), (lBot.x, lBot.y)) +
                   math.dist((rTop.x, rTop.y), (rBot.x, rBot.y))) / 2.0
        return avgDist < 0.018

    def getIrisPosition(self, landmarks):
        center = landmarks[468]
        rightC = landmarks[33]
        leftC = landmarks[133]
        width = math.dist((leftC.x, leftC.y), (rightC.x, rightC.y))
        distRight = math.dist((center.x, center.y), (rightC.x, rightC.y))
        ratioX = distRight / width

        topLid = landmarks[159]
        botLid = landmarks[145]
        height = math.dist((topLid.x, topLid.y), (botLid.x, botLid.y))
        distTop = math.dist((center.x, center.y), (topLid.x, topLid.y))
        ratioY = distTop / height
        return ratioX, ratioY

    def update(self, landmarks):
        self.eyesClosed = self.getEyeOpenness(landmarks)
        rX, rY = self.getIrisPosition(landmarks)
        targetX = winWidth / 2 + (rX - 0.52) * winWidth * sensitivity * 6.0
        targetY = winHeight / 2 + (rY - 0.45) * winHeight * sensitivity * 10.0
        targetX = max(50, min(winWidth - 50, targetX))
        targetY = max(50, min(winHeight - 50, targetY))
        self.currentX += (targetX - self.currentX) * smoothFactor
        self.currentY += (targetY - self.currentY) * smoothFactor
        return pg.Vector2(self.currentX, self.currentY), self.eyesClosed


# --- 5. UI Drawing Functions ---

def drawCyberHUD(surface, isGathering):
    panelW, panelH = 340, 150
    panelX, panelY = 20, 20
    panelRect = pg.Rect(panelX, panelY, panelW, panelH)
    pg.draw.rect(surface, colorBlackTrans, panelRect, border_radius=15)
    pg.draw.rect(surface, colorCyan, panelRect, 2, border_radius=15)

    txtTitleCN = fontTitleBig.render("眼神控制系统", True, colorWhite)
    txtTitleEN = fontTitleSub.render("EYE CONTROL SYSTEM // V5.0", True, colorCyan)
    surface.blit(txtTitleCN, (panelX + 20, panelY + 15))
    surface.blit(txtTitleEN, (panelX + 22, panelY + 48))
    pg.draw.line(surface, (*colorWhite, 100), (panelX + 20, panelY + 70), (panelX + panelW - 20, panelY + 70), 1)

    statusStr = "状态: 聚能中" if isGathering else "状态: 游离态"
    statusColor = colorRed if isGathering else colorGreen
    pg.draw.circle(surface, statusColor, (panelX + 280, panelY + 35), 8)
    if isGathering:
        glowSize = 12 + math.sin(pg.time.get_ticks() * 0.01) * 3
        surfGlow = pg.Surface((40, 40), pg.SRCALPHA)
        pg.draw.circle(surfGlow, (*statusColor, 100), (20, 20), glowSize)
        surface.blit(surfGlow, (panelX + 260, panelY + 15), special_flags=pg.BLEND_ADD)

    bullet1 = fontBody.render("● 闭眼 / 眨眼  >>>  聚气充能", True, colorPink)
    bullet2 = fontBody.render("● 瞬间睁眼    >>>  释放爆发", True, colorYellow)
    surface.blit(bullet1, (panelX + 20, panelY + 80))
    surface.blit(bullet2, (panelX + 20, panelY + 110))


# --- [新功能] 炫酷眼球追踪框 ---
def drawEyeTracker(surface, faceLms, aimPoint):
    # 1. 获取左眼球坐标 (Landmark 468 是左虹膜中心)
    iris = faceLms[468]
    cx, cy = int(iris.x * winWidth), int(iris.y * winHeight)

    # 颜色随时间微变，增加科技感
    hue = (pg.time.get_ticks() % 2000) / 2000.0
    # 这里简单用青色，你可以加上HSV变换
    trackColor = colorCyan

    # 2. 绘制旋转锁定环 (Segments)
    radius = 28
    time = pg.time.get_ticks() * 0.005  # 旋转速度

    # 画三段弧线模拟旋转的UI
    for i in range(3):
        startAng = time + i * (2 * math.pi / 3)
        endAng = startAng + 1.5  # 弧长
        pg.draw.arc(surface, trackColor, (cx - radius, cy - radius, radius * 2, radius * 2), startAng, endAng, 2)

    # 3. 绘制边角锁定框 (Brackets)
    bracketSize = 35
    gap = 8
    length = 10

    # 左上
    pg.draw.lines(surface, trackColor, False,
                  [(cx - bracketSize, cy - bracketSize + length), (cx - bracketSize, cy - bracketSize),
                   (cx - bracketSize + length, cy - bracketSize)], 2)
    # 右上
    pg.draw.lines(surface, trackColor, False,
                  [(cx + bracketSize - length, cy - bracketSize), (cx + bracketSize, cy - bracketSize),
                   (cx + bracketSize, cy - bracketSize + length)], 2)
    # 左下
    pg.draw.lines(surface, trackColor, False,
                  [(cx - bracketSize, cy + bracketSize - length), (cx - bracketSize, cy + bracketSize),
                   (cx - bracketSize + length, cy + bracketSize)], 2)
    # 右下
    pg.draw.lines(surface, trackColor, False,
                  [(cx + bracketSize - length, cy + bracketSize), (cx + bracketSize, cy + bracketSize),
                   (cx + bracketSize, cy + bracketSize - length)], 2)

    # 4. 绘制数据连线 (Data Link)
    # 从眼睛连线到目标点，用虚线或半透明线
    pg.draw.line(surface, (*trackColor, 80), (cx, cy), (aimPoint.x, aimPoint.y), 1)

    # 5. 核心点
    pg.draw.circle(surface, (*colorWhite, 200), (cx, cy), 3)


# --- 6. MainLoop ---
controlSystem = GazeAndBlinkSystem()
for _ in range(maxOrbs):
    orbList.append(CyberOrb())

aimPoint = pg.Vector2(winWidth // 2, winHeight // 2)
isGatheringState = False
lastGatheringState = False

appRunning = True

while appRunning:
    ticks = pg.time.get_ticks()
    for ev in pg.event.get():
        if ev.type == pg.QUIT:
            appRunning = False

    # A. Vision
    hasFrame, rawFrame = camSource.read()
    if not hasFrame: continue

    rawFrame = cv.convertScaleAbs(rawFrame, alpha=0.7, beta=-20)
    mirrorFrame = cv.flip(rawFrame, 1)
    rgbData = cv.cvtColor(mirrorFrame, cv.COLOR_BGR2RGB)
    results = faceTracker.process(rgbData)

    videoSurf = pg.surfarray.make_surface(np.transpose(rgbData, (1, 0, 2)))
    mainCanvas.blit(videoSurf, (0, 0))

    fxLayer = pg.Surface((winWidth, winHeight), pg.SRCALPHA)

    # B. Control
    if results.multi_face_landmarks:
        faceLms = results.multi_face_landmarks[0].landmark
        aimPoint, isGatheringState = controlSystem.update(faceLms)

        if lastGatheringState == True and isGatheringState == False:
            for orb in orbList:
                orb.applyImpulse(aimPoint, burstForce)
        lastGatheringState = isGatheringState

        # [新增] 绘制眼球追踪炫酷小框
        # 传入 fxLayer 以支持透明度
        drawEyeTracker(fxLayer, faceLms, aimPoint)

    # C. Particles
    for orb in orbList:
        orb.update(aimPoint, isGatheringState)
        orb.draw(fxLayer, ticks)

    # D. Cursor
    cursorColor = colorPink if isGatheringState else colorCyan
    pg.draw.circle(fxLayer, (*cursorColor, 100), (int(aimPoint.x), int(aimPoint.y)), 20, 2)
    pg.draw.circle(fxLayer, (*colorWhite, 200), (int(aimPoint.x), int(aimPoint.y)), 4)

    mainCanvas.blit(fxLayer, (0, 0), special_flags=pg.BLEND_ADD)

    # E. UI
    drawCyberHUD(mainCanvas, isGatheringState)

    pg.display.flip()
    appClock.tick(60)

camSource.release()
pg.quit()